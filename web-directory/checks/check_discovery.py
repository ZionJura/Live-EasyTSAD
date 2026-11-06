from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import os
from pathlib import Path
import tempfile
import threading
import unittest

spec = importlib.util.spec_from_file_location('discover', Path(__file__).parents[1] / 'discover.py')
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)


class DiscoveryChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / 'index.html').write_text('<title>Real Project</title><meta name="description" content="A useful project">')
        self.config = {'public_host': 'example.com', 'public_ip': '192.0.2.1', 'projects': []}

    def tearDown(self):
        self.temp.cleanup()

    def route(self, port=80, public=True, protected=False, root=True):
        return {'url': f'http://example.com:{port}/demo/', 'port': port, 'path': '/demo/',
                'root': str(self.root) if root else None, 'index': str(self.root / 'index.html') if root else None,
                'public': public, 'protected': protected, 'backend': None}

    def test_listener_parsing_ipv6_and_infrastructure_exclusion(self):
        text = '\n'.join(['LISTEN 0 511 127.0.0.1:8766 0.0.0.0:* users:(("nginx",pid=1,fd=1))',
                          'LISTEN 0 511 [::]:443 [::]:* users:(("nginx",pid=1,fd=2))',
                          'LISTEN 0 511 0.0.0.0:22 0.0.0.0:* users:(("sshd",pid=2,fd=3))',
                          'LISTEN 0 511 *:9090 *:* users:(("mihomo",pid=3,fd=4))'])
        result = d.read_listeners(text, dict(self.config, exclude_ports=[22], exclude_processes=['mihomo']))
        self.assertEqual(set(result), {443, 8766})
        self.assertEqual(result[443]['addresses'], ['::'])

    def test_includes_and_protected_alias_and_proxy_resolve(self):
        source = f'''# configuration file /etc/nginx/nginx.conf:
http {{ include /etc/nginx/conf.d/*.conf; }}
# configuration file /etc/nginx/conf.d/site.conf:
server {{ listen 80; server_name example.com; include /etc/nginx/snippets/routes.conf; }}
server {{ listen 127.0.0.1:8766; root "{self.root}"; index index.html; location / {{ try_files $uri =404; }} }}
# configuration file /etc/nginx/snippets/routes.conf:
location ^~ /demo/ {{ proxy_pass http://127.0.0.1:8766/; }}
location /private/ {{ auth_request /_authorize; alias "{self.root}/"; }}
location = /_authorize {{ internal; proxy_pass http://127.0.0.1:9999; }}
'''
        routes = d.candidates(d.nginx_tree(source), self.config, {80: {}, 8766: {}})
        self.assertEqual(len(routes), 3)
        demo = next(c for c in routes if c['path'] == '/demo/')
        private = next(c for c in routes if c['path'] == '/private/')
        self.assertEqual(demo['root'], str(self.root))
        self.assertTrue(private['protected'])
        self.assertEqual(demo['backend'], 8766)

    def test_project_dedup_uses_public_route_without_bypassing_auth(self):
        private, internal = self.route(protected=True), self.route(port=8766, public=False)
        probes = {private['url']: {'status': 302}, internal['url']: {'status': 200, 'html': True}}
        catalog, _ = d.build_catalog(self.config, {}, [private, internal], probes, {}, 100)
        self.assertEqual(len(catalog['projects']), 1)
        project = catalog['projects'][0]
        self.assertEqual(project['url'], private['url'])
        self.assertTrue(project['protected'])
        self.assertNotIn('useful', project['description'])
        self.assertNotEqual(project['title'], 'Real Project')
        self.assertEqual(project['ports'], [80, 8766])

    def test_file_timestamp_and_sort_do_not_use_scan_time(self):
        first = self.route()
        secondroot = self.root / 'newer'
        secondroot.mkdir()
        (secondroot / 'index.html').write_text('<title>Newer</title>')
        # Nested assets should contribute, logs should not.
        (secondroot / 'app.js').write_text('hello')
        for path in [self.root / 'index.html', secondroot / 'index.html']:
            os.utime(path, (100, 100))
        os.utime(secondroot / 'app.js', (200, 200))
        (secondroot / 'debug.log').write_text('ignored')
        second = dict(first, root=str(secondroot), index=str(secondroot / 'index.html'), url='http://example.com/new/')
        catalog, _ = d.build_catalog(self.config, {}, [second], {}, {}, 100000)
        self.assertEqual(catalog['projects'][0]['modifiedAt'], d.iso(200))
        self.assertEqual(catalog['projects'][0]['modifiedSource'], 'files')

    def test_unknown_and_observed_content_change(self):
        route = self.route(root=False, public=False)
        initial = {route['url']: {'html': True, 'status': 200, 'fingerprint': 'a'}}
        catalog, state = d.build_catalog(self.config, {}, [route], initial, {}, 100)
        self.assertIsNone(catalog['projects'][0]['modifiedAt'])
        self.assertIsNone(catalog['projects'][0]['url'])
        catalog, state = d.build_catalog(self.config, {}, [route], initial, state, 200)
        self.assertIsNone(catalog['projects'][0]['modifiedAt'])
        changed = {route['url']: {'html': True, 'status': 200, 'fingerprint': 'b'}}
        catalog, state = d.build_catalog(self.config, {}, [route], changed, state, 300)
        self.assertEqual(catalog['projects'][0]['modifiedAt'], d.iso(300))
        self.assertEqual(catalog['projects'][0]['modifiedSource'], 'observed')
        catalog, _ = d.build_catalog(self.config, {}, [route], changed, state, 400)
        self.assertEqual(catalog['projects'][0]['modifiedAt'], d.iso(300))

    def test_loopback_listener_is_not_published(self):
        catalog, _ = d.build_catalog(self.config, {8080: {'addresses': ['127.0.0.1']}}, [],
                                    {'port:8080': {'url': 'http://localhost:8080/', 'html': True, 'status': 200,
                                                   'title': 'Local app'}}, {}, 100)
        project = catalog['projects'][0]
        self.assertIsNone(project['url'])
        self.assertEqual(project['visibility'], 'internal')
        self.assertEqual(catalog['summary']['htmlPorts'], [8080])

    def test_public_listener_gets_direct_link_without_proxy(self):
        catalog, _ = d.build_catalog(self.config, {8080: {'addresses': ['0.0.0.0']}}, [],
                                    {'port:8080': {'url': 'http://localhost:8080/', 'html': True, 'status': 200}}, {}, 100)
        project = catalog['projects'][0]
        self.assertEqual(project['url'], 'http://192.0.2.1:8080/')
        self.assertEqual(project['accessMode'], 'port')
        self.assertEqual(project['status'], 'online')

    def test_json_proxy_is_not_a_web_project(self):
        route = self.route(root=False)
        catalog, _ = d.build_catalog(self.config, {}, [route], {route['url']: {'status': 200, 'html': False}}, {}, 100)
        self.assertEqual(catalog['projects'], [])

    def test_probe_rejects_json_and_error_html_and_does_not_follow_redirects(self):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                status, kind, content = {
                    '/html': (200, 'text/html', '<title>Test &amp; app</title>'),
                    '/json': (200, 'application/json', '{}'),
                    '/missing': (404, 'text/html', '<title>Missing</title>'),
                    '/redirect': (302, 'text/html', ''),
                }[self.path]
                self.send_response(status)
                self.send_header('Content-Type', kind)
                self.send_header('Location', '/html')
                self.end_headers()
                self.wfile.write(content.encode())

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f'http://localhost:{server.server_port}'
            self.assertEqual(d.probe(url + '/html')['title'], 'Test & app')
            self.assertTrue(d.probe(url + '/html')['html'])
            for path in ('/json', '/missing', '/redirect'):
                self.assertFalse(d.probe(url + path)['html'])
        finally:
            server.shutdown()
            server.server_close()


if __name__ == '__main__':
    unittest.main()
