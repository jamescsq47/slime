"""Fail closed before launching GPU workers if keyless search is unusable."""

import argparse
import asyncio
import json
from pathlib import Path

from data.bfcl_web_search.tools import WebTools, WebBackendError


async def check(backend):
    tools = WebTools(backend=backend, timeout=15)
    probes = [("Marie Curie first Nobel prize year", ["curie", "1903"]),
              ("Super Bowl 2024 halftime performer", ["usher"])]
    passed = True
    for query, expected in probes:
        try:
            results = await tools.call({"name": "search_engine_query", "arguments": {"keywords": query, "max_results": 5}})
            content = json.dumps(results).lower()
            relevant = all(word in content for word in expected)
            passed &= relevant
            print(json.dumps({"query": query, "relevance_check": relevant, "results": results}, ensure_ascii=False))
        except WebBackendError as exc:
            passed = False
            print(json.dumps({"query": query, "error": str(exc)}))
    print(json.dumps({"passed": passed, "events": tools.events}, ensure_ascii=False))
    try:
        page = await tools.call({"name": "fetch_url_content", "arguments": {
            "url": "https://en.wikipedia.org/wiki/Marie_Curie", "mode": "truncate"}})
        fetch_ok = "Curie" in page["content"] and len(page["content"]) > 500
    except WebBackendError:
        fetch_ok = False
    print(json.dumps({"passed": passed and fetch_ok, "fetch_ok": fetch_ok, "events": tools.events}, ensure_ascii=False))
    return passed and fetch_ok


if __name__ == "__main__":
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    backend = config["datasets"][0]["options"]["search_backend"]
    raise SystemExit(0 if asyncio.run(check(backend)) else 1)
