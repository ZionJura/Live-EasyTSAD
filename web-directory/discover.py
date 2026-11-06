#!/usr/bin/env python3
"""Read local listeners and Nginx routes; publish metadata, never new proxies.

Python standard library + ss + nginx + curl. No external scanner or service needed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import fnmatch
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlsplit


def run(args, timeout=15):
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        # Keep diagnostic errors in the private service journal, never in the web JSON.
        print(result.stderr[:4000], file=sys.stderr)
        result.check_returncode()
    return result.stdout


def iso(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat().replace("+00:00", "Z")


class Metadata(HTMLParser):
    def __init__(self):
        super().__init__()
        self.title = ""
        self.description = ""
        self.in_title = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "title":
            self.in_title = True
        if tag == "meta" and (attrs.get("name", "").lower() == "description"
                              or attrs.get("property", "").lower() == "og:description"):
            self.description = attrs.get("content", "")

    def handle_endtag(self, tag):
        if tag == "title":
            self.in_title = False

    def handle_data(self, value):
        if self.in_title:
            self.title += value


def metadata(body):
    parser = Metadata()
    parser.feed(body[:262144])
    clean = lambda s, limit: re.sub(r"\s+", " ", s).strip()[:limit]
    return clean(parser.title, 100), clean(parser.description, 240)


def parse_nginx(source):
    """Small directive/block parser. Quoted strings and comments stay intact."""
    tokens = re.findall(r'''\#[^\n]*|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|[{};]|[^\s{};#"']+''', source)
    tokens = [t[1:-1] if t[:1] in ('"', "'") else t for t in tokens if not t.startswith('#')]
    index = 0

    def block():
        nonlocal index
        result, words = [], []
        while index < len(tokens):
            token = tokens[index]
            index += 1
            if token == '}':
                return result
            if token == ';':
                if words:
                    result.append((words, []))
                words = []
            elif token == '{':
                result.append((words, block()))
                words = []
            else:
                words.append(token)
        return result
    return block()


def nginx_tree(dump):
    chunks = re.split(r'^# configuration file (.+):\s*$', dump, flags=re.M)
    files = {chunks[i]: parse_nginx(chunks[i + 1]) for i in range(1, len(chunks), 2)}

    def expand(nodes, stack):
        result = []
        for words, children in nodes:
            if words and words[0] == 'include' and len(words) > 1:
                for path, included in files.items():
                    if fnmatch.fnmatch(path, words[1]) and path not in stack:
                        result.extend(expand(included, stack | {path}))
            else:
                result.append((words, expand(children, stack)))
        return result
    first = next(iter(files), None)
    return expand(files.get(first, []), {first})


def values(nodes, name):
    return [words[1:] for words, _ in nodes if words and words[0] == name]


def one(nodes, name, default=None):
    found = values(nodes, name)
    return found[-1][0] if found and found[-1] else default


def servers(tree):
    for words, children in tree:
        if words == ['http']:
            for args, body in children:
                if args == ['server']:
                    yield body


def listen_address(value):
    if value.startswith('unix:'):
        return None
    if value.isdigit():
        return '0.0.0.0', int(value)
    if ':' in value:
        host, port = value.rsplit(':', 1)
        if port.isdigit():
            return host.strip('[]'), int(port)
    return None


def loopback(host):
    return host in ('localhost', '::1') or host.startswith('127.')


def read_listeners(text, config):
    entries = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 4:
            continue
        address = listen_address(fields[3])
        if not address:
            continue
        host, port = address
        processes = re.findall(r'\("([^"]+)"', line)
        if port in config.get('exclude_ports', []) or any(
                any(p.startswith(excluded) for excluded in config.get('exclude_processes', [])) for p in processes):
            continue
        current = entries.setdefault(port, {'port': port, 'addresses': [], 'processes': processes})
        if host not in current['addresses']:
            current['addresses'].append(host)
    return entries


def excluded_root(root, config):
    return root and any(root == item or root.startswith(item.rstrip('/') + '/')
                        for item in config.get('exclude_roots', []))


def index_file(root, indexes=None):
    if not root or '$' in root:
        return None
    for name in indexes or ['index.html', 'index.htm']:
        candidate = Path(root) / name
        if candidate.is_file() and candidate.suffix in ('.html', '.htm'):
            return candidate
    return None


def candidates(tree, config, listeners):
    result = []
    for body in servers(tree):
        listens = [args for args in values(body, 'listen') if args and listen_address(args[0])]
        hostnames = [name for args in values(body, 'server_name') for name in args
                     if re.fullmatch(r'[a-zA-Z0-9.-]+', name) and name not in ('localhost', '_')]
        preferred = config['public_host']
        host = preferred if preferred in hostnames else next(iter(hostnames), config['public_ip'])
        root = one(body, 'root')
        indexes = next(iter(values(body, 'index')), ['index.html', 'index.htm'])
        locations = [(words, child) for words, child in body if words and words[0] == 'location']
        if not any(words[-1] == '/' for words, _ in locations):
            locations.append((['location', '/'], []))
        for words, child in locations:
            path = words[-1]
            if path in config.get('exclude_paths', []) or not path.startswith('/') or not path.endswith('/'):
                continue
            if len(words) > 2 and words[1] in ('~', '~*', '='):
                continue
            if values(child, 'return') or values(child, 'internal'):
                continue
            proxy = one(child, 'proxy_pass')
            alias = one(child, 'alias')
            local_root = alias or one(child, 'root', root)
            if local_root and not alias and path != '/' and not proxy:
                local_root = str(Path(local_root) / path.lstrip('/'))
            page = index_file(local_root, next(iter(values(child, 'index')), indexes))
            if not proxy and (not page or excluded_root(local_root, config)):
                continue
            if any('$' in value for value in (proxy or '', path, host)):
                continue
            protected = any(one(child, key, one(body, key, 'off')) != 'off'
                            for key in ('auth_request', 'auth_basic', 'auth_jwt'))
            backend = None
            if proxy:
                parsed = urlsplit(proxy)
                if not loopback(parsed.hostname or ''):
                    continue
                backend = parsed.port or (443 if parsed.scheme == 'https' else 80)
            for listen in listens:
                address, port = listen_address(listen[0])
                if port not in listeners:
                    continue
                public = not loopback(address)
                scheme = 'https' if 'ssl' in listen else 'http'
                target_host = host if public else 'localhost'
                netloc = target_host + (f':{port}' if port != (443 if scheme == 'https' else 80) else '')
                result.append({'url': f'{scheme}://{netloc}{path}', 'port': port, 'path': path,
                               'root': str(Path(local_root)) if page else None,
                               'index': str(page) if page else None, 'public': public,
                               'protected': protected, 'backend': backend})
    # A mounted route and its loopback preview refer to the same root/project.
    by_port = {c['port']: c for c in result if not c['public'] and c['root'] and c['path'] == '/'}
    for candidate in result:
        backend = by_port.get(candidate['backend'])
        if backend:
            candidate['root'], candidate['index'] = backend['root'], backend['index']
    return result


def probe(url, connect_host='127.0.0.1'):
    """Local GET only, no redirects or environment proxy. Preserve HTTPS SNI."""
    parsed = urlsplit(url)
    port = parsed.port or (443 if parsed.scheme == 'https' else 80)
    target = f'[{connect_host}]' if ':' in connect_host else connect_host
    with tempfile.TemporaryDirectory() as tmp:
        headers, body = Path(tmp) / 'headers', Path(tmp) / 'body'
        command = ['curl', '--noproxy', '*', '--silent', '--max-time', '3', '--connect-timeout', '2',
                   '--max-filesize', '1048576', '--connect-to', f'{parsed.hostname}:{port}:{target}:{port}',
                   '--dump-header', str(headers), '--output', str(body), '--write-out', '%{http_code}',
                   '--user-agent', 'ProjectDirectory/1.0', url]
        response = subprocess.run(command, capture_output=True, text=True, timeout=5)
        status = int(response.stdout) if response.stdout.isdigit() else 0
        header_text = headers.read_text(errors='replace') if headers.exists() else ''
        header_map = {k.lower(): v.strip() for k, v in re.findall(r'^([^:\r\n]+):([^\r\n]*)', header_text, re.M)}
        content = body.read_bytes()[:262144].decode('utf-8', errors='replace') if body.exists() else ''
        is_html = 200 <= status < 300 and any(kind in header_map.get('content-type', '').lower()
                                             for kind in ('text/html', 'application/xhtml+xml'))
        title, description = metadata(content) if is_html else ('', '')
        modified = None
        try:
            modified = parsedate_to_datetime(header_map['last-modified']).timestamp()
        except (KeyError, TypeError, ValueError, OverflowError):
            pass
        return {'status': status, 'html': is_html, 'title': title, 'description': description,
                'last_modified': modified, 'fingerprint': hashlib.sha256(content.encode()).hexdigest() if is_html else None,
                'redirect': header_map.get('location', '')}


ASSET_EXTENSIONS = {'.html', '.htm', '.css', '.js', '.mjs', '.json', '.svg', '.png', '.webp',
                    '.jpg', '.jpeg', '.ico', '.woff', '.woff2', '.pdf', '.md'}


def file_modified(root):
    latest = None
    count = 0
    for directory, subdirs, files in os.walk(Path(root).resolve(), followlinks=False):
        subdirs[:] = [d for d in subdirs if not d.startswith('.') and d not in ('node_modules', 'logs', '__pycache__', 'releases')]
        for name in files:
            if Path(name).suffix.lower() not in ASSET_EXTENSIONS:
                continue
            path = Path(directory) / name
            if path.is_symlink():
                continue
            try:
                stamp = path.stat().st_mtime
                latest = max(latest or stamp, stamp)
            except OSError:
                pass
            count += 1
            if count >= 20000:
                return latest
    return latest


def public_choice(c):
    parsed = urlsplit(c['url'])
    return (c['public'], parsed.scheme == 'https', c['port'] in (80, 443), c['path'] != '/')


def build_catalog(config, listeners, routes, probes, previous, now):
    groups = {}
    for candidate in routes:
        if excluded_root(candidate['root'], config):
            continue
        key = 'root:' + str(Path(candidate['root']).resolve()) if candidate['root'] else 'port:' + str(candidate['backend'] or candidate['port'])
        # Keep stable identity when an atomic deployment switches the current symlink.
        if candidate['root']:
            key = 'root:' + candidate['root'].rstrip('/')
        groups.setdefault(key, []).append(candidate)
    represented = {c['port'] for c in routes} | {c['backend'] for c in routes if c['backend']}
    for port, listener in listeners.items():
        if port in represented:
            continue
        entry = probes.get(f'port:{port}', {})
        if entry.get('html'):
            public = any(not loopback(a) for a in listener['addresses'])
            scheme = urlsplit(entry['url']).scheme
            host = config['public_host'] if scheme == 'https' else config['public_ip']
            url = f'{scheme}://{host}:{port}/' if public else entry['url']
            # This only links an existing listener; it never opens a port or adds a proxy.
            groups[f'port:{port}'] = [{'url': url, 'port': port, 'root': None, 'index': None,
                                      'path': '/', 'protected': False, 'public': public, 'backend': None,
                                      'direct_port': public}]
    projects, state = [], {}
    for key, group in groups.items():
        if not any(c['index'] or probes.get(c['url'], probes.get(f"port:{c['port']}", {})).get('html') for c in group):
            # A proxy location may serve JSON, metrics, or an API, rather than a webpage.
            continue
        best = max(group, key=public_choice)
        root = best['root']
        override = next((item for item in config.get('projects', [])
                         if (root and item.get('root', '').rstrip('/') == root.rstrip('/'))
                         or item.get('port') in [c['port'] for c in group]), {})
        if override.get('hidden'):
            continue
        pid = hashlib.sha256(key.encode()).hexdigest()[:12]
        old = previous.get(pid, {})
        result = probes.get(best['url'], probes.get(f"port:{best['port']}", {}))
        available = any(probes.get(c['url'], probes.get(f"port:{c['port']}", {})).get('html') for c in group)
        protected = best['protected'] or result.get('status') in (401, 403)
        # Don't follow a login redirect or substitute its page metadata for a project.
        protected = protected or bool(re.search(r'/(login|signin|root)(/|\?|$)', result.get('redirect', '')))
        title, description = result.get('title', ''), result.get('description', '')
        if best['index'] and not protected:
            try:
                title, description = metadata(Path(best['index']).read_text(errors='replace'))
            except OSError:
                pass
        title = override.get('title') or (title if not protected else '') or (Path(root).parent.name if root else f'网页服务 · {best["port"]}')
        description = override.get('description') or (description if not protected else '') or (
            '此项目需要登录后访问。' if protected else '页面未提供简介，可在目录配置中补充。')
        stamp = file_modified(root) if root else None
        source = 'files' if stamp is not None else None
        if stamp is None and result.get('last_modified') is not None:
            stamp, source = result['last_modified'], 'http'
        fingerprint = result.get('fingerprint') or old.get('fingerprint')
        if stamp is None and old.get('fingerprint') and fingerprint != old['fingerprint']:
            stamp, source = now, 'observed'
        elif stamp is None and old.get('modified_source') == 'observed':
            stamp, source = old.get('modified'), 'observed'
        online = (result.get('html') if best['public'] else available) or (protected and result.get('status', 0) in (200, 301, 302, 303, 307, 308, 401, 403))
        ports = sorted({c['port'] for c in group} | {c['backend'] for c in group if c['backend']})
        projects.append({'id': pid, 'title': title, 'description': description,
                         'category': override.get('category', '网页应用' if best['public'] else '内部服务'),
                         'icon': override.get('icon', 'window'), 'url': best['url'] if best['public'] else None,
                         'path': best['path'] if best['public'] else None, 'ports': ports,
                         'visibility': 'published' if best['public'] else 'internal',
                         'accessMode': 'port' if best.get('direct_port') else 'route',
                         'protected': protected, 'status': 'online' if online else 'offline',
                         'modifiedAt': iso(stamp) if stamp is not None else None,
                         'modifiedSource': source, 'firstSeenAt': old.get('first_seen', iso(now))})
        state[pid] = {'first_seen': old.get('first_seen', iso(now)), 'fingerprint': fingerprint,
                      'modified': stamp, 'modified_source': source}
    projects.sort(key=lambda p: (p['modifiedAt'] or '', p['title']), reverse=True)
    html_ports = sorted({int(key.split(':')[1]) for key, value in probes.items() if key.startswith('port:') and value.get('html')}
                        | {c['port'] for c in routes if probes.get(c['url'], {}).get('html')})
    return {'generatedAt': iso(now), 'refreshSeconds': 60, 'projects': projects,
            'summary': {'projects': len(projects), 'published': sum(p['visibility'] == 'published' for p in projects),
                        'internal': sum(p['visibility'] == 'internal' for p in projects),
                        'htmlPortCount': len(html_ports), 'htmlPorts': html_ports,
                        'online': sum(p['status'] == 'online' for p in projects)}}, state


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
            f.write('\n')
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def discover(config):
    listeners = read_listeners(run(['ss', '-H', '-ltnp']), config)
    tree = nginx_tree(run(['nginx', '-T']))
    routes = candidates(tree, config, listeners)
    jobs = {c['url']: (c['url'], '127.0.0.1') for c in routes}
    for port, listener in listeners.items():
        scheme = 'https' if port in (443, 8443) else 'http'
        host = config['public_host'] if scheme == 'https' else 'localhost'
        addresses = listener['addresses']
        address = '127.0.0.1' if any(a in ('0.0.0.0', '127.0.0.1', '*', '::') for a in addresses) else addresses[0]
        jobs[f'port:{port}'] = (f'{scheme}://{host}:{port}/', address)

    def execute(item):
        key, (url, address) = item
        try:
            result = probe(url, address)
            if key.startswith('port:') and result['status'] == 0 and not url.startswith('https:'):
                # A TLS-only application may listen on a nonstandard port.
                tls_url = f'https://{config["public_host"]}:{urlsplit(url).port}/'
                result = probe(tls_url, address)
                if result['html']:
                    url = tls_url
            return key, dict(result, url=url)
        except (OSError, subprocess.TimeoutExpired):
            return key, {'status': 0, 'html': False, 'url': url}
    with ThreadPoolExecutor(max_workers=8) as executor:
        probes = dict(executor.map(execute, jobs.items()))
    return listeners, routes, probes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--state', required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    previous = json.loads(Path(args.state).read_text()) if Path(args.state).exists() else {}
    listeners, routes, probes = discover(config)
    catalog, state = build_catalog(config, listeners, routes, probes, previous, time.time())
    atomic_json(args.state, state)
    atomic_json(args.output, catalog)
    print(json.dumps(catalog['summary'], ensure_ascii=False))


if __name__ == '__main__':
    main()
