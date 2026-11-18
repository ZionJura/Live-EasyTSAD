---
name: fetch-repository
description: Locate the paper and official source repository for a TSAD reproduction, produce the source selection consumed by the controller retrieval scripts.
---

Resolve the requested paper and official repository from the paper, author/project pages and repository documentation. Record source URLs and the selected commit. Use the supplied search/fetch tools to inspect candidates; a similarly named repository is not evidence of authorship.

Write only `reproduction/source.json` with `repository_url`, `supported`, `evidence` (source URLs or paper references supporting official authorship), and a `paper` object with the discovered `title`, `url`, `pdf_url` and `arxiv_id` when available. Use `find_papers`, `find_repositories` and `fetch_url` to investigate a title or project URL when the initial material is missing. Optional `search_web` uses the configured SearXNG instance. The controller downloads the selected PDF and clones the verified repository into the assigned method's `upstream/`; it records commit/tree provenance. Inspect available material rather than returning an unverified URL. Do not clone/download a second copy or install dependencies from the discovery worker.

Return `unsupported` with the search evidence when no official repository can be found. Do not synthesize a replacement algorithm, choose an unrelated implementation, or claim that a URL alone completes FETCH. Do not start training.
