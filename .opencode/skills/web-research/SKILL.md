---
name: web-research
description: Search the web with DuckDuckGo and fetch page content. Use when answering questions needing current docs, library APIs, reference implementations, or fact-checking. Prefers MCP tools over raw websearch.
---

# Web Research

Use the project's connected MCP servers (`duckduckgo`, `fetch`) before falling back to anything else.

## Workflow

1. Search first with `duckduckgo_search` (5-10 results, then narrow region only if results are locale-sensitive).
2. Expand shortened `ref://` tokens with `duckduckgo_expand_link` when present.
3. Fetch the 2-3 most relevant pages with `duckduckgo_fetch_content` (preferred, strips nav/chrome) or `fetch_fetch` (raw fallback, max_length 8000).
4. Cite the source URL for every factual claim taken from the web.

## Rules

- Never fabricate URLs. Only cite pages actually returned by search or fetch.
- If a page blocks scraping, try the next result instead of guessing its contents.
- For Hugging Face repos, prefer the `huggingface` MCP tools (`hub_repo_search`, `hub_repo_details`) over web fetch.
- For arXiv papers, prefer the `arxiv` MCP tools (`get_abstract`) before downloading.
