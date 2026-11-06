"""Module 1: paper search, resolution and download (the FETCH phase).

* `search(query)`  - candidate papers from the arXiv Atom API, optional GitHub
  repository search (code links) and an LLM web-search fallback through the
  configured OpenAI Responses endpoint.
* `resolve(...)`   - one paper record from an arXiv id, a title, a URL or a local PDF.
* `fetch(task)`    - download the PDF, convert it to `paper/paper.md` (PyMuPDF,
  `<!-- page N -->` markers), find and shallow-clone the official code into
  `code/<repo>`, and write `paper/meta.json`. `fetch_to(dir, record)` is the
  task-independent core.

HTTP goes through `urllib` (the bundled httpx 1.0 dev build has no proxy
support); for every request the proxy is chosen from `config.network_env()`, so
direct hosts (arXiv, the LLM endpoint) bypass the proxy and GitHub uses it.
"""
from __future__ import annotations

import difflib
import http.client
import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

from .config import HarnessError, load
from .task import Task, now

log = logging.getLogger(__name__)

KEYS = ("title", "authors", "year", "venue", "arxiv_id", "url", "pdf_url", "abstract", "code_urls", "source")
UA = "tsad-harness/0.1"
ATOM = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
ARXIV_ID_RE = re.compile(r"(?<![\d.])(\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})(v\d+)?", re.I)
GITHUB_RE = re.compile(r"github\.com/([A-Za-z0-9_.\-]+)/([A-Za-z0-9_.\-]+)(?:/tree/([A-Za-z0-9_.\-]+))?")
GITHUB_NON_REPO = {"features", "topics", "orgs", "about", "pricing", "marketplace", "sponsors", "settings", "login"}


# ---------------------------------------------------------------- networking
def _proxies_for(url: str) -> dict:
    """Proxy map for one URL: empty when the host is in NO_PROXY of network_env()."""
    env = load().network_env()
    host = urllib.parse.urlsplit(url).hostname or ""
    no_proxy = env.get("no_proxy") or env.get("NO_PROXY") or ""
    if urllib.request.proxy_bypass_environment(host, {"no": no_proxy}):
        return {}
    proxies = {}
    for scheme in ("http", "https"):
        val = env.get(f"{scheme}_proxy") or env.get(f"{scheme.upper()}_PROXY")
        if val:
            proxies[scheme] = val
    return proxies


def http(url: str, *, data: bytes | None = None, headers: dict | None = None,
         timeout: float = 60.0, retries: int = 3) -> bytes:
    """GET (or POST when `data` is given) with proxy rules from network_env and retries on 429/5xx."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler(_proxies_for(url)))
    req = urllib.request.Request(url, data=data, headers={"User-Agent": UA, **(headers or {})})
    for attempt in range(retries):
        try:
            with opener.open(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == retries - 1:
                body = e.read()[:500].decode("utf-8", "replace")
                raise HarnessError(f"HTTP {e.code} for {url}: {body}") from e
        except (urllib.error.URLError, http.client.HTTPException, OSError) as e:
            if attempt == retries - 1:
                raise HarnessError(f"request failed for {url}: {e}") from e
        time.sleep(3 * (attempt + 1))
    raise AssertionError("unreachable")


# --------------------------------------------------------------------- arXiv
_arxiv_lock = threading.Lock()
_arxiv_last = 0.0


def _arxiv_query(**params) -> list[dict]:
    """Call export.arxiv.org/api/query (spaced >= 3 s apart, as arXiv asks) and parse the Atom feed."""
    global _arxiv_last
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(params)
    with _arxiv_lock:
        time.sleep(max(0.0, 3.0 - (time.time() - _arxiv_last)))
        try:
            xml = http(url, timeout=60)
        finally:
            _arxiv_last = time.time()
    return [_atom_entry(e) for e in ET.fromstring(xml).findall("a:entry", ATOM)
            if "/abs/" in e.findtext("a:id", "", ATOM)]


def _atom_entry(e: ET.Element) -> dict:
    txt = lambda tag: " ".join((e.findtext(tag, "", ATOM) or "").split())
    m = ARXIV_ID_RE.search(txt("a:id").split("/abs/")[-1])
    aid = m.group(1) if m else ""
    abstract = txt("a:summary")
    pdf = next((l.get("href") for l in e.findall("a:link", ATOM) if l.get("title") == "pdf"), "")
    return record(
        title=txt("a:title"), authors=[txt_ for a in e.findall("a:author", ATOM)
                                       if (txt_ := " ".join(a.findtext("a:name", "", ATOM).split()))],
        year=int(txt("a:published")[:4] or 0) or None,
        venue=txt("arxiv:journal_ref") or "arXiv", arxiv_id=aid,
        url=f"https://arxiv.org/abs/{aid}", pdf_url=pdf or f"https://arxiv.org/pdf/{aid}",
        abstract=abstract, code_urls=github_links(abstract + " " + txt("arxiv:comment")), source="arxiv")


def arxiv_search(query: str, limit: int = 10, *, field: str = "all") -> list[dict]:
    """Relevance-sorted arXiv search; `field="ti"` searches the title as a phrase."""
    words = re.findall(r"[\w\-]+", query)
    if not words:
        return []
    q = f'ti:"{" ".join(words)}"' if field == "ti" else " AND ".join(f"all:{w}" for w in words)
    return _arxiv_query(search_query=q, start=0, max_results=limit, sortBy="relevance")


def parse_arxiv_id(text: str) -> str | None:
    """Extract a bare arXiv id ("2110.02642") from an id, `arXiv:` string or abs/pdf URL."""
    m = ARXIV_ID_RE.search(text.split("arxiv.org/")[-1] if "arxiv.org/" in text else text)
    return m.group(1) if m else None


# -------------------------------------------------------------------- GitHub
def github_links(text: str) -> list[str]:
    """Unique `https://github.com/<owner>/<repo>[/tree/<branch>]` links in order of appearance."""
    out = []
    text = re.sub(r"(github\.com/\S*[-_/])\s+(?!https?:|github)(?=\w)", r"\1", text or "")   # URL wrapped by the PDF layout
    for owner, repo, branch in GITHUB_RE.findall(text):
        repo = re.sub(r"(\.git|[.\-_]+)$", "", repo)
        if not repo or owner.lower() in GITHUB_NON_REPO:
            continue
        url = f"https://github.com/{owner}/{repo}" + (f"/tree/{branch}" if branch else "")
        if not any(u.lower().split("/tree/")[0] == url.lower().split("/tree/")[0] for u in out):
            out.append(url)
    return out


def github_search(query: str, limit: int = 5) -> list[dict]:
    """GitHub repository search (unauthenticated unless $GITHUB_TOKEN is set)."""
    headers = {"Accept": "application/vnd.github+json"}
    if os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
    url = "https://api.github.com/search/repositories?" + urllib.parse.urlencode({"q": query, "per_page": limit})
    items = json.loads(http(url, headers=headers, retries=1)).get("items", [])
    return [{"url": i["html_url"], "full_name": i["full_name"], "stars": i.get("stargazers_count", 0),
             "description": i.get("description") or ""} for i in items]


def find_code(title: str, limit: int = 3) -> list[str]:
    """Best-match repositories for the short title (the part before ":"); unverified candidates."""
    words = " ".join(re.findall(r"[\w\-]+", title.split(":")[0]))
    return [r["url"] for r in github_search(f'"{words}" in:name,description,readme', limit)] if words else []


# ----------------------------------------------------------------------- LLM
def llm_search(query: str, limit: int = 5) -> list[dict]:
    """Web search through the configured Responses API; the model answers with a JSON list."""
    cfg = load()
    llm = cfg["llm"]
    if llm.get("wire_api", "responses") != "responses":
        log.warning("llm search skipped: wire_api=%r has no web_search tool", llm.get("wire_api"))
        return []
    prompt = (
        f"Find up to {limit} research papers on time-series anomaly detection matching: {query!r}.\n"
        "Use web search. Prefer the original paper and its official code repository.\n"
        "Answer with ONLY a JSON array (no prose, no markdown fences). Each element: "
        '{"title": str, "authors": [str], "year": int, "venue": str, "arxiv_id": str ("" if none), '
        '"url": str, "pdf_url": str (direct PDF link or ""), "abstract": str (<= 400 chars), '
        '"code_urls": [str] (official repositories only, [] if unknown)}.')
    body = json.dumps({"model": llm["model"], "input": prompt, "tools": [{"type": "web_search"}]}).encode()
    raw = http(llm["base_url"].rstrip("/") + "/responses", data=body, timeout=600, retries=2,
               headers={"Authorization": f"Bearer {cfg.api_key()}", "Content-Type": "application/json"})
    text = "\n".join(c.get("text", "") for item in json.loads(raw).get("output", [])
                     if item.get("type") == "message" for c in item.get("content", []) or [])
    return [record(**{**{k: v for k, v in c.items() if k in KEYS}, "source": "llm"})
            for c in _json_list(text)[:limit] if isinstance(c, dict) and c.get("title")]


def _json_list(text: str) -> list:
    """Parse the first JSON array in a model answer (tolerates fences and surrounding prose)."""
    text = re.sub(r"```(?:json)?", "", text)
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end <= start:
        return []
    try:
        val = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return []
    return val if isinstance(val, list) else []


# ------------------------------------------------------------------- records
def record(**kw) -> dict:
    """Normalized paper record with every key in KEYS (extra keys such as `pdf_path` are kept)."""
    r = {"title": "", "authors": [], "year": None, "venue": "", "arxiv_id": "", "url": "",
         "pdf_url": "", "abstract": "", "code_urls": [], "source": "", **kw}
    r["title"] = " ".join(str(r["title"] or "").split())
    if isinstance(r["authors"], str):
        r["authors"] = [a.strip() for a in re.split(r",| and ", r["authors"]) if a.strip()]
    r["authors"] = r["authors"] or []
    try:
        r["year"] = int(r["year"]) if r["year"] else None
    except (TypeError, ValueError):
        r["year"] = None
    r["arxiv_id"] = parse_arxiv_id(str(r["arxiv_id"] or "")) or ""
    if r["arxiv_id"]:
        r["url"] = r["url"] or f"https://arxiv.org/abs/{r['arxiv_id']}"
        r["pdf_url"] = r["pdf_url"] or f"https://arxiv.org/pdf/{r['arxiv_id']}"
    r["code_urls"] = github_links(" ".join(r["code_urls"] or [])) if isinstance(r["code_urls"], list) else []
    return r


def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


def _merge(dst: dict, src: dict) -> dict:
    """Fill empty fields of `dst` from `src` and union the code links."""
    for k, v in src.items():
        if k == "code_urls":
            dst[k] = github_links(" ".join(dst.get(k, []) + v))
        elif not dst.get(k) and v:
            dst[k] = v
    return dst


def search(query: str, *, limit: int = 10, sources: tuple[str, ...] = ("arxiv", "llm")) -> list[dict]:
    """Candidate papers from the given sources ("arxiv", "llm", "github"), deduplicated.

    "github" does not add papers; it fills `code_urls` of the top candidates that
    have none (one GitHub search per candidate, unauthenticated limit 10/min).
    """
    out: list[dict] = []

    def add(recs: list[dict]) -> None:
        for r in recs:
            key_t, key_a = _norm_title(r["title"]), r["arxiv_id"]
            hit = next((o for o in out if (key_a and o["arxiv_id"] == key_a) or _norm_title(o["title"]) == key_t), None)
            _merge(hit, r) if hit else out.append(r)

    for src, fn in (("arxiv", lambda: arxiv_search(query, limit)), ("llm", lambda: llm_search(query, min(limit, 8)))):
        if src in sources:
            try:
                add(fn())
            except Exception as e:  # one failing source must not hide the others
                log.warning("%s search failed: %s", src, e)
    if "github" in sources:
        for r in [r for r in out if not r["code_urls"]][:3]:
            try:
                r["code_urls"] = find_code(r["title"])
            except Exception as e:
                log.warning("github search failed: %s", e)
                break
    return out[:limit]


def resolve(*, arxiv: str | None = None, title: str | None = None,
            pdf: Path | str | None = None, url: str | None = None) -> dict:
    """One paper record. Priority: arXiv id (also parsed from an arxiv.org URL) > title > URL > local PDF."""
    arxiv = parse_arxiv_id(arxiv) if arxiv else (parse_arxiv_id(url) if url and "arxiv.org" in url else None)
    rec: dict | None = None
    if arxiv:
        hits = _arxiv_query(id_list=arxiv, max_results=1)
        if not hits:
            raise HarnessError(f"arXiv id {arxiv} not found")
        rec = hits[0]
    elif title:
        rec = _best_title_match(title, arxiv_search(title, 5, field="ti") or arxiv_search(title, 10))
        if rec is None:
            try:
                rec = _best_title_match(title, llm_search(title, 3), cutoff=0.7)
            except Exception as e:
                log.warning("llm search failed: %s", e)
        if rec is None:
            rec = record(title=title, source="manual")
    if url and not arxiv:
        link = {"code_urls": github_links(url)} if "github.com" in url else (
            {"pdf_url": url} if url.lower().split("?")[0].endswith(".pdf") or "/pdf" in url else {"url": url})
        rec = _merge(rec or record(source="url"), record(**link))
    if pdf:
        p = Path(pdf).expanduser().resolve()
        if not p.is_file():
            raise HarnessError(f"no PDF at {p}")
        rec = rec or record(title=_pdf_title(p), source="local")
        rec["pdf_path"] = str(p)
    if rec is None:
        raise HarnessError("resolve() needs one of arxiv=, title=, url=, pdf=")
    return rec


def _best_title_match(title: str, recs: list[dict], cutoff: float = 0.85) -> dict | None:
    scored = [(difflib.SequenceMatcher(None, _norm_title(title), _norm_title(r["title"])).ratio(), r) for r in recs]
    best = max(scored, key=lambda s: s[0], default=(0.0, None))
    return best[1] if best[0] >= cutoff else None


# ----------------------------------------------------------------------- PDF
def _pdf_title(path: Path) -> str:
    import pymupdf
    with pymupdf.open(path) as doc:
        t = (doc.metadata or {}).get("title") or ""
        if not t.strip() and len(doc):
            t = next((l for l in doc[0].get_text().splitlines() if len(l.strip()) > 10), "")
    return " ".join(t.split()) or path.stem


def pdf_to_markdown(path: Path, title: str = "") -> tuple[str, dict]:
    """PDF -> readable text with `<!-- page N -->` markers; also returns {pages, links}.

    Text blocks with long lines (prose) are joined into one paragraph with
    hyphenation removed; short-line blocks (tables, equations) keep their breaks.
    """
    import pymupdf
    parts = [f"# {title}\n"] if title else []
    links: list[str] = []
    with pymupdf.open(path) as doc:
        for i, page in enumerate(doc, 1):
            links += [l["uri"] for l in page.get_links() if l.get("uri")]
            parts.append(f"\n<!-- page {i} -->\n")
            for b in page.get_text("blocks"):
                if b[6] != 0:          # image block
                    continue
                lines = [" ".join(l.split()) for l in b[4].splitlines()]
                lines = [l for l in lines if l]
                if not lines:
                    continue
                if sum(map(len, lines)) / len(lines) > 40:
                    para = ""
                    for l in lines:
                        para = para[:-1] + l if re.search(r"[a-z]-$", para) and l[:1].islower() else (
                            f"{para} {l}" if para else l)
                    parts.append(para + "\n")
                else:
                    parts.append("\n".join(lines) + "\n")
        pages = len(doc)
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(parts)).strip() + "\n"
    return text, {"pages": pages, "links": list(dict.fromkeys(links))}


def _download_pdf(url: str, dest: Path) -> None:
    data = http(url, timeout=180, headers={"Accept": "application/pdf"})
    if not data.startswith(b"%PDF"):
        raise HarnessError(f"{url} did not return a PDF ({len(data)} bytes, starts {data[:20]!r})")
    dest.write_bytes(data)


# ---------------------------------------------------------------------- code
def clone(url: str, dest_root: Path) -> dict:
    """`git clone --depth 1` a GitHub link into dest_root/<repo>; reuses an existing clone."""
    m = GITHUB_RE.search(url)
    if not m:
        raise HarnessError(f"not a GitHub repository URL: {url}")
    owner, repo, branch = m.group(1), re.sub(r"\.git$", "", m.group(2)), m.group(3)
    dest = dest_root / repo
    env = load().network_env()
    if not (dest / ".git").exists():
        if dest.exists():
            shutil.rmtree(dest)
        cmd = ["git", "clone", "--depth", "1", "--quiet", *(["--branch", branch] if branch else []),
               f"https://github.com/{owner}/{repo}.git", str(dest)]
        p = subprocess.run(cmd, env={**env, "GIT_TERMINAL_PROMPT": "0"}, capture_output=True, text=True, timeout=900)
        if p.returncode != 0:
            raise HarnessError(f"git clone failed: {p.stderr.strip()[-500:]}")
    commit = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"], capture_output=True,
                            text=True, env=env).stdout.strip()
    return {"url": f"https://github.com/{owner}/{repo}", "branch": branch or "", "dir": str(dest), "commit": commit}


# --------------------------------------------------------------------- fetch
def fetch_to(root: Path, rec: dict, *, code_url: str = "", clone_code: bool = True) -> dict:
    """Populate root/paper (paper.pdf, paper.md, meta.json) and root/code/<repo>; return the meta dict."""
    paper_dir, code_dir = root / "paper", root / "code"
    paper_dir.mkdir(parents=True, exist_ok=True)
    code_dir.mkdir(parents=True, exist_ok=True)
    rec = record(**rec)
    meta: dict = {"fetched_at": now(), "paper": rec, "pdf": None, "markdown": None,
                  "code": None, "code_candidates": [], "errors": []}
    pdf = paper_dir / "paper.pdf"
    if rec.get("pdf_path"):
        shutil.copyfile(rec["pdf_path"], pdf)
    elif rec["pdf_url"] and not (pdf.exists() and pdf.read_bytes()[:4] == b"%PDF"):   # keep it on a re-run
        _download_pdf(rec["pdf_url"], pdf)
    elif not rec["pdf_url"]:
        raise HarnessError("paper record has neither pdf_url nor pdf_path")
    md, info = pdf_to_markdown(pdf, rec["title"])
    (paper_dir / "paper.md").write_text(md)
    meta["pdf"] = {"path": str(pdf), "bytes": pdf.stat().st_size, "pages": info["pages"]}
    meta["markdown"] = {"path": str(paper_dir / "paper.md"), "chars": len(md)}

    # code: explicit url > links in the record/abstract > PDF hyperlinks > PDF text
    found = github_links(" ".join(rec["code_urls"] + info["links"]) + " " + md)
    meta["code_candidates"] = found[:10]
    target, source = (code_url, "task") if code_url else ((found[0], "paper") if found else ("", ""))
    if not target and rec["title"]:
        try:
            meta["code_candidates"] = find_code(rec["title"])
            source = "github-search (not cloned: unverified)"
        except Exception as e:
            meta["errors"].append(f"github search: {e}")
    if target and clone_code:
        try:
            meta["code"] = {**clone(target, code_dir), "source": source}
        except Exception as e:
            meta["errors"].append(f"clone {target}: {e}")
    elif target:
        meta["code"] = {"url": target, "source": source, "dir": None, "commit": None}
    meta["code_source"] = source
    (paper_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    return meta


def fetch(task: Task) -> dict:
    """FETCH phase for a task: resolve the paper if needed, then `fetch_to(task.dir, ...)`."""
    rec = dict(task.spec.get("paper") or {})
    if not (rec.get("pdf_url") or rec.get("pdf_path")):
        rec = _merge(resolve(arxiv=rec.get("arxiv_id") or None, title=rec.get("title") or None,
                             url=rec.get("url") or None), rec)
        task.log("fetch/resolved", paper=rec)
    meta = fetch_to(task.dir, rec, code_url=task.spec.get("code_url", ""))
    task.log("fetch/done", pdf=meta["pdf"], markdown=meta["markdown"], code=meta["code"],
             code_candidates=meta["code_candidates"], errors=meta["errors"])
    return meta
