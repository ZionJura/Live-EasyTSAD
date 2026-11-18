"""Scoped MCP tools over stdio or per-turn loopback HTTP for flow workers."""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
import tomllib
import urllib.request
from pathlib import Path

from .config import ROOT, HarnessError
from .task import Task
from .workers import is_write_allowed, relative


def resolve_path(value: str, *, allowed_roots: tuple[Path, ...] = ()) -> Path:
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise HarnessError("Use a repository-relative path without '..'")
    root = ROOT.resolve()
    lexical = root / candidate
    path = lexical.resolve()
    if not path.is_relative_to(root):
        raise HarnessError("Path resolves outside the repository")
    for checked in (lexical, path):
        parts = checked.relative_to(root).parts
        dependency = re.fullmatch(r"envs/[^/]+/venv/(?:pyvenv\.cfg|lib/python\d+\.\d+/site-packages(?:/.*)?)",
                                  checked.relative_to(root).as_posix()) is not None
        if not dependency and any(checked.is_relative_to(scope / "environment/.venv") for scope in allowed_roots):
            dependency = "/site-packages/" in checked.as_posix() or checked.name in ("site-packages", "pyvenv.cfg")
        if any((part.startswith(".") and not (part == ".venv" and dependency))
               or part in ("node_modules", "__pycache__", "access-token")
               or (part == "venv" and not dependency) for part in parts):
            raise HarnessError("Credential and hidden files are unavailable to workers")
        if checked.is_relative_to(root / "runtime") and not any(checked.is_relative_to(scope) for scope in allowed_roots):
            raise HarnessError("Runtime files outside this task and method are unavailable to workers")
    return path


def _read_file(path: str, start_line: int = 1, max_lines: int = 300,
               *, allowed_roots: tuple[Path, ...] = ()) -> str:
    source = resolve_path(path, allowed_roots=allowed_roots)
    if source.suffix == ".npy":
        import numpy as np
        values = np.load(source, mmap_mode="r", allow_pickle=False)
        metadata = {"shape": list(values.shape), "dtype": str(values.dtype),
                    "preview": values.reshape(-1)[:8].tolist()}
        if "label" in source.stem.lower() and values.size <= 1_000_000:
            metadata["unique_values"] = np.unique(values).tolist()
        return json.dumps(metadata)
    if source.stat().st_size > 5_000_000:
        raise HarnessError("Use a smaller source file")
    lines = source.read_text(errors="replace").splitlines()
    start = max(0, start_line - 1)
    return "\n".join(f"{n + 1}: {line}" for n, line in enumerate(lines[start:start + min(max_lines, 1000)], start))


def _list_files(path: str, pattern: str = "*", *, allowed_roots: tuple[Path, ...] = ()) -> str:
    root = resolve_path(path, allowed_roots=allowed_roots)
    repository = ROOT.resolve()
    files = []
    pending = [root]
    deadline = time.monotonic() + 30
    while pending:
        if time.monotonic() >= deadline:
            raise HarnessError("File listing exceeded 30 seconds; select a narrower source directory.")
        parent = pending.pop()
        try:
            resolve_path(parent.relative_to(repository).as_posix(), allowed_roots=allowed_roots)
            readable = True
        except HarnessError:
            readable = False
        with os.scandir(parent) as entries:
            for entry in entries:
                if time.monotonic() >= deadline:
                    raise HarnessError("File listing exceeded 30 seconds; select a narrower source directory.")
                if entry.name.startswith(".") or entry.name in ("access-token", "venv", "node_modules", "__pycache__"):
                    continue
                item = Path(entry.path)
                if entry.is_dir(follow_symlinks=False):
                    if allowed_roots and parent == allowed_roots[0] and entry.name in ("guard", "dsh", "code"):
                        continue
                    if readable or any(scope.is_relative_to(item) or item.is_relative_to(scope) for scope in allowed_roots):
                        pending.append(item)
                    continue
                if not readable or not item.match(pattern):
                    continue
                filename = item.relative_to(repository).as_posix()
                if entry.is_symlink():
                    try:
                        resolved = resolve_path(filename, allowed_roots=allowed_roots)
                    except HarnessError:
                        continue
                    if not resolved.is_file():
                        continue
                elif not entry.is_file(follow_symlinks=False):
                    continue
                files.append(filename)
                if len(files) >= 1000:
                    return json.dumps(files)
    return json.dumps(files)


def _search_files(path: str, pattern: str, *, allowed_roots: tuple[Path, ...] = ()) -> str:
    root = resolve_path(path, allowed_roots=allowed_roots)
    command = ["rg", "--json", "--max-count", "20", "--glob", "!uv.lock", "--glob", "!*.jsonl"]
    if not root.is_relative_to(ROOT.resolve() / "runtime"):
        command += ["--glob", "!/runtime/**"]
    result = subprocess.run([*command, "--", pattern, relative(root)], cwd=ROOT,
                            capture_output=True, text=True, timeout=30)
    matches = []
    for line in result.stdout.splitlines():
        event = json.loads(line)
        if event.get("type") != "match":
            continue
        data = event["data"]
        name = data["path"].get("text")
        text = data["lines"].get("text")
        if name is None or text is None:
            continue
        try:
            resolve_path(name, allowed_roots=allowed_roots)
        except HarnessError:
            continue
        matches.append(f"{name}:{data['line_number']}:{text.rstrip(chr(10))}")
    return "\n".join(matches)[:40000]


def write_file(task, role: str, path: str, content: str) -> dict:
    allowed_roots = (task.dir.absolute(), task.method_dir.absolute())
    target = resolve_path(path, allowed_roots=allowed_roots)
    lexical = ROOT.resolve() / path
    if not is_write_allowed(task, role, lexical):
        raise HarnessError(f"{role} cannot write {path}")
    target.parent.mkdir(parents=True, exist_ok=True)
    # A parent symlink may have appeared after path validation.
    if not is_write_allowed(task, role, lexical) or not is_write_allowed(task, role, target):
        raise HarnessError("Write path changed during validation")
    target.write_text(content)
    return {"path": relative(target), "bytes": len(content.encode())}


def main(*, check_only=False, http_port=None):
    if http_port is not None and (not isinstance(http_port, int) or not 1 <= http_port <= 65535):
        raise HarnessError("HTTP port must be between 1 and 65535")
    http_path = os.environ.get("TSAD_WORKER_MCP_HTTP_PATH", "/mcp")
    if not re.fullmatch(r"/mcp(?:/[0-9a-f]{32})?", http_path):
        raise HarnessError("HTTP path must be /mcp or /mcp/<32 hexadecimal characters>")
    try:
        from mcp.server.mcpserver import MCPServer as FastMCP
        run_http_options = True
    except ImportError:
        from mcp.server.fastmcp import FastMCP
        run_http_options = False
    directory = Path(os.environ["TSAD_TASK_DIR"])
    with open(directory / "task.toml", "rb") as source:
        task = Task(directory, tomllib.load(source), json.loads((directory / "state.json").read_text()))
    role = os.environ["TSAD_WORKER_ROLE"]
    # This server handles one task for its entire lifetime; avoid re-reading
    # task/config files for every candidate in a recursive source listing.
    allowed_roots = (task.dir.absolute(), task.method_dir.absolute())
    options = ({"host": "127.0.0.1", "port": http_port, "streamable_http_path": http_path}
               if http_port is not None and not run_http_options else {})
    server = FastMCP("tsad-worker", log_level="WARNING", **options)
    tool = server.tool(structured_output=False)

    @tool
    def read_file(path: str, start_line: int = 1, max_lines: int = 300) -> str:
        """Read text with line numbers, or NPY shape/dtype/preview and label values."""
        return _read_file(path, start_line, max_lines, allowed_roots=allowed_roots)

    @tool
    def list_files(path: str, pattern: str = "*") -> str:
        """List source files underneath a repository-relative directory."""
        return _list_files(path, pattern, allowed_roots=allowed_roots)

    @tool
    def search_files(path: str, pattern: str) -> str:
        """Search source with a regular expression; returns repository-relative lines."""
        return _search_files(path, pattern, allowed_roots=allowed_roots)

    @tool
    def publish_file(path: str, content: str) -> str:
        """Publish an assigned artifact or a permitted method-local file."""
        if os.environ.get("TSAD_WORKER_READ_ONLY") == "1":
            raise HarnessError("Session continuation is read-only; submit a job correction to request changes")
        return json.dumps(write_file(task, role, path, content))

    # Keep the prompt-facing name explicit and unambiguous.
    publish_file.__doc__ = "write_file: publish within the role's assigned artifact or method-directory scope."

    if role == "fetcher":
        @tool
        def find_papers(query: str) -> str:
            """Find paper metadata and PDF links through arXiv (no model call)."""
            from .papers import search
            return json.dumps(search(query, limit=8, sources=("arxiv",)))

        @tool
        def search_web(query: str) -> str:
            """Search the configured SearXNG JSON provider for paper/source evidence."""
            from .config import load
            from .papers import http
            from urllib.parse import urlencode
            endpoint = load().raw.get("search", {}).get("base_url", "")
            if not endpoint:
                raise HarnessError("SearXNG is not configured; use find_papers, find_repositories or fetch_url")
            url = endpoint + ("&" if "?" in endpoint else "?") + urlencode({"q": query, "format": "json"})
            result = json.loads(http(url, timeout=45, retries=1))
            return json.dumps({"results": result.get("results", [])[:10]}, ensure_ascii=False)

        @tool
        def find_repositories(title: str) -> str:
            """Find repository candidates. Verify author/paper evidence before selection."""
            from .papers import find_code
            return json.dumps(find_code(title))

        @tool
        def fetch_url(url: str) -> str:
            """Read a public paper/repository webpage to verify official provenance."""
            from urllib.parse import urlparse
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
                raise HarnessError("Expected a public HTTP URL")
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "TSAD-research/1"}), timeout=45) as response:
                return response.read(250000).decode(errors="replace")

    if check_only:
        print(json.dumps({"ready": True, "role": role}))
        return
    if http_port is not None:
        if run_http_options:
            server.run("streamable-http", host="127.0.0.1", port=http_port,
                       streamable_http_path=http_path)
        else:
            server.run("streamable-http")
        return
    server.run("stdio")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Initialize scoped MCP tools without starting the transport")
    parser.add_argument("--http-port", type=int, help="Serve scoped tools over loopback streamable HTTP")
    arguments = parser.parse_args()
    main(check_only=arguments.check, http_port=arguments.http_port)
