#!/usr/bin/env python3
"""Update the deduplicated citation count used by the static personal site."""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, quote
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "citation_data.json"
CONTACT = "tangjw24@mails.tsinghua.edu.cn"

# These are the stable identifiers currently linked from the Publications section.
PUBLICATIONS = [
    {"title": "GMSA: Enhancing Context Compression via Group Merging and Layer Semantic Alignment", "arxiv": "2505.12215"},
    {"title": "COMI: Coarse-to-Fine Context Compression via Marginal Information Gain", "arxiv": "2602.01719"},
    {"title": "Read As Human: Compressing Context via Parallelizable Close Reading and Skimming", "arxiv": "2602.01840"},
    {"title": "Perception Compressor: A Training-Free Prompt Compression Framework in Long Context Scenarios", "arxiv": "2409.19272"},
    {"title": "CoS: Towards Optimal Event Scheduling via Chain-of-Scheduling", "arxiv": "2511.12913"},
    {"title": "Beyond Position Bias: Shifting Context Compression from Position-Driven to Semantic-Driven", "arxiv": "2605.09463"},
]


def get_json(url: str, params: dict[str, str] | None = None, retries: int = 4) -> Any:
    if params:
        separator = "&" if "?" in url else "?"
        url = url + separator + urlencode(params)
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": f"Twilightaaa.github.io citation updater (mailto:{CONTACT})",
        },
    )
    for attempt in range(retries):
        try:
            with urlopen(request, timeout=40) as response:
                return json.load(response)
        except (HTTPError, URLError, TimeoutError) as exc:
            if attempt == retries - 1:
                raise RuntimeError(f"request failed: {url}: {exc}") from exc
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKD", value or "").lower()
    return re.sub(r"[^a-z0-9]+", "", value)


def normalize_doi(value: str | None) -> str | None:
    if not value:
        return None
    value = value.lower().strip()
    value = re.sub(r"^https?://(dx\.)?doi\.org/", "", value)
    return value.rstrip(".")


def arxiv_from_url(value: str | None) -> str | None:
    if not value:
        return None
    match = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9.]+)", value, re.I)
    return match.group(1) if match else None


def citation_key(item: dict[str, Any], source: str) -> str:
    ids = item.get("ids") or item.get("externalIds") or {}
    doi = normalize_doi(ids.get("doi") or item.get("doi"))
    if doi:
        return f"doi:{doi}"
    arxiv = ids.get("arxiv") or ids.get("ArXiv") or arxiv_from_url(item.get("url"))
    if arxiv:
        return f"arxiv:{arxiv.lower()}"
    pmid = ids.get("pmid") or ids.get("PubMed")
    if pmid:
        return f"pmid:{pmid}"
    title = normalize(item.get("title") or "")
    year = str(item.get("publication_year") or item.get("year") or "")
    if title:
        return f"title:{title}:{year}"
    # This is only a last resort. It prevents two malformed records from
    # collapsing into one while still keeping the source record identifiable.
    return f"{source}:{item.get('id') or item.get('paperId') or repr(item)}"


def openalex_citations(arxiv: str) -> list[dict[str, Any]]:
    work = get_json(f"https://api.openalex.org/works/https://arxiv.org/abs/{quote(arxiv)}", {"mailto": CONTACT})
    cited_by = work.get("cited_by_api_url")
    if not cited_by:
        return []
    records: list[dict[str, Any]] = []
    cursor = "*"
    # 50 pages per work is enough for this profile and prevents a runaway job.
    for _ in range(50):
        page = get_json(cited_by, {"per-page": "200", "cursor": cursor, "mailto": CONTACT})
        records.extend(page.get("results", []))
        cursor = (page.get("meta") or {}).get("next_cursor")
        if not cursor or not page.get("results"):
            break
        time.sleep(0.1)
    return records


def semanticscholar_citations(arxiv: str) -> list[dict[str, Any]]:
    headers_url = f"https://api.semanticscholar.org/graph/v1/paper/ARXIV:{quote(arxiv)}"
    params = {"fields": "paperId,title,year,externalIds,url"}
    headers = {"Accept": "application/json", "User-Agent": f"citation updater (mailto:{CONTACT})"}
    # Resolve the paper ID first, because the citations endpoint is keyed by it.
    request = Request(headers_url + "?" + urlencode(params), headers=headers)
    with urlopen(request, timeout=40) as response:
        paper = json.load(response)
    paper_id = paper.get("paperId")
    if not paper_id:
        return []
    records: list[dict[str, Any]] = []
    offset = 0
    for _ in range(100):
        query = {"fields": "paperId,title,year,externalIds,url", "limit": "100", "offset": str(offset)}
        url = f"https://api.semanticscholar.org/graph/v1/paper/{quote(paper_id)}/citations?{urlencode(query)}"
        request = Request(url, headers=headers)
        with urlopen(request, timeout=40) as response:
            page = json.load(response)
        batch = [entry.get("citingPaper", {}) for entry in page.get("data", [])]
        records.extend(batch)
        if len(batch) < 100:
            break
        offset += len(batch)
        time.sleep(0.2)
    return records


def read_existing() -> dict[str, Any]:
    try:
        return json.loads(OUTPUT.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {"status": "pending"}


def main() -> int:
    all_citations: dict[str, dict[str, Any]] = {}
    work_summaries: list[dict[str, Any]] = []
    successful_sources = 0
    failures: list[str] = []

    for publication in PUBLICATIONS:
        arxiv = publication["arxiv"]
        openalex_records: list[dict[str, Any]] = []
        s2_records: list[dict[str, Any]] = []
        try:
            openalex_records = openalex_citations(arxiv)
            successful_sources += 1
        except Exception as exc:  # Keep one unavailable index from blocking all others.
            failures.append(f"OpenAlex {arxiv}: {exc}")
        try:
            s2_records = semanticscholar_citations(arxiv)
            successful_sources += 1
        except Exception as exc:
            failures.append(f"Semantic Scholar {arxiv}: {exc}")

        work_keys: set[str] = set()
        for source, records in (("openalex", openalex_records), ("semantic_scholar", s2_records)):
            for record in records:
                key = citation_key(record, source)
                work_keys.add(key)
                all_citations.setdefault(
                    key,
                    {"title": record.get("title"), "year": record.get("publication_year") or record.get("year"), "sources": set()},
                )["sources"].add(source)
        work_summaries.append({"title": publication["title"], "arxiv": arxiv, "deduplicated_citations": len(work_keys)})

    # Do not replace a previously valid count with an empty result when both
    # indexes are temporarily unavailable or rate-limited.
    if successful_sources == 0:
        print("No citation index responded; preserving the previous result.", file=sys.stderr)
        return 1

    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    output = {
        "total_citations": len(all_citations),
        "last_updated": today,
        "status": "ok",
        "sources": ["OpenAlex", "Semantic Scholar"],
        "deduplication": "Union of citing works matched by DOI, arXiv, PubMed, or normalized title and publication year.",
        "works": work_summaries,
    }
    if failures:
        output["warnings"] = failures
    OUTPUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote {len(all_citations)} deduplicated citations to {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
