---
name: browser-automation
description: Drive a real Chromium browser via the Playwright MCP server for JS-rendered pages, visual verification, and frontend smoke tests. Use when fetch returns truncated content, the page needs JS, or verifying the React frontend.
---

# Browser Automation

Requires Node 24 + Playwright Chromium + the `playwright` MCP server in `opencode.json`.

## Setup

```bash
export PATH="$PATH:/c/Program Files/nodejs"
npx -y playwright install chromium
opencode mcp list
```

Expected `opencode.json` entry:

```json
{
  "mcp": {
    "servers": {
      "playwright": {
        "type": "local",
        "command": ["C:/Program Files/nodejs/node.exe", "C:/Users/aakif/AppData/Roaming/npm/node_modules/@playwright/mcp/cli.js"]
      }
    }
  }
}
```

Absolute paths are used because the background service starts with a stale
`PATH` that predates the Node.js install; `npx` is not resolved there.

## Workflow

1. Prefer `duckduckgo_fetch_content` / `fetch_fetch` for static pages (cheaper, faster).
2. Switch to Playwright only when the page needs JS rendering, login interaction, or visual confirmation.
3. Scope each session to one task, then close the browser to free resources.
4. For React frontend verification (`src/frontend/react-app`), drive `http://localhost:3002` after `docker compose up`.

## Rules

- Never use the browser for bulk scraping; fetch one page per step.
- Do not enter credentials unless the user explicitly provides them for that session.
