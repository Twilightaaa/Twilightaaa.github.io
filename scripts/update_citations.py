#!/usr/bin/env python3
"""Refresh the site's deduplicated citation aggregate.

The preferred order mirrors the aggregation used by zjsxply.github.io:
Google Scholar is the baseline, then Semantic Scholar and ADS contribute only
previously unseen citing works. Stable identifiers are preferred; exact title
and year matching is used only when no stable identifier is available.

When the optional Google Scholar/ADS credentials are not configured, OpenAlex
is used as the baseline so the scheduled job remains useful on a fresh repo.
"""

from __future__ import annotations

import datetime as dt
import difflib
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote, urlencode, urlparse
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "citation_data.json"
CONTACT = "tangjw24@mails.tsinghua.edu.cn"
SERPAPI_KEY = os.environ.get("SERPAPI_API_KEY", "").strip()
ADS_TOKEN = os.environ.get("ADS_API_TOKEN", "").strip()
S2_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "").strip()
try:
    MANUAL_SCHOLAR_BASELINE = int(os.environ.get("GOOGLE_SCHOLAR_BASELINE", "168"))
except ValueError:
    MANUAL_SCHOLAR_BASELINE = 168

PUBLICATIONS = [
    {"title": "GMSA: Enhancing Context Compression via Group Merging and Layer Semantic Alignment", "arxiv": "2505.12215"},
    {"title": "COMI: Coarse-to-Fine Context Compression via Marginal Information Gain", "arxiv": "2602.01719"},
    {"title": "Read As Human: Compressing Context via Parallelizable Close Reading and Skimming", "arxiv": "2602.01840"},
    {"title": "Perception Compressor: A Training-Free Prompt Compression Framework in Long Context Scenarios", "arxiv": "2409.19272"},
    {"title": "CoS: Towards Optimal Event Scheduling via Chain-of-Scheduling", "arxiv": "2511.12913"},
    {"title": "Beyond Position Bias: Shifting Context Compression from Position-Driven to Semantic-Driven", "arxiv": "2605.09463"},
]


def get_json(
    url: str,
    params: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
    retries: int = 4,
) -> Any:
    if params:
        separator = "&" if "?" in url else "?"
        url = url + separator + urlencode(params)
    request_headers = {
        "Accept": "application/json",
        "User-Agent": f"Twilightaaa.github.io citation updater (mailto:{CONTACT})",
    }
    request_headers.update(headers or {})
    request = Request(url, headers=request_headers)
    for attempt in range(retries):
        try:
            with urlopen(request, timeout=40) as response:
                return json.load(response)
        except (HTTPError, URLError, TimeoutError) as exc:
            if attempt == retries - 1:
                raise RuntimeError(f"request failed: {url}: {exc}") from exc
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def normalize_title(value: str | None) -> str:
    value = unicodedata.normalize("NFKD", value or "").lower().strip()
    value = re.sub(r"^\s*\[[^]]+\]\s*", "", value)
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


def arxiv_from_identifiers(value: Any) -> str | None:
    if isinstance(value, list):
        for item in value:
            found = arxiv_from_identifiers(item)
            if found:
                return found
        return None
    if not isinstance(value, str):
        return None
    match = re.search(r"(?:arxiv:|arxiv\.org/(?:abs|pdf)/)([0-9.]+)", value, re.I)
    return match.group(1) if match else None


def doi_from_url(value: str | None) -> str | None:
    if not value:
        return None
    match = re.search(r"doi\.org/(10\.\d{4,9}/[-._;()/:a-z0-9]+)", value, re.I)
    return normalize_doi(match.group(1)) if match else None


def year_from(value: Any) -> str | None:
    if value is None:
        return None
    match = re.search(r"\b(19|20)\d{2}\b", str(value))
    return match.group(0) if match else None


def canonical_record(
    title: str | None,
    year: Any = None,
    link: str | None = None,
    doi: str | None = None,
    arxiv: str | None = None,
    pmid: str | None = None,
    source_id: str | None = None,
) -> dict[str, Any]:
    return {
        "title": (title or "").strip(),
        "year": year_from(year),
        "link": link or "",
        "ids": {
            "doi": normalize_doi(doi) or doi_from_url(link),
            "arxiv": arxiv or arxiv_from_url(link),
            "pmid": pmid,
            "source": source_id,
        },
    }


def record_identities(record: dict[str, Any]) -> set[tuple[str, str]]:
    ids = record.get("ids") or {}
    identities: set[tuple[str, str]] = set()
    if ids.get("doi"):
        identities.add(("doi", str(ids["doi"]).lower()))
    if ids.get("arxiv"):
        identities.add(("arxiv", str(ids["arxiv"]).lower()))
    if ids.get("pmid"):
        identities.add(("pmid", str(ids["pmid"]).lower()))
    return identities


def record_title_key(record: dict[str, Any]) -> tuple[str, str] | None:
    title = normalize_title(record.get("title"))
    if not title:
        return None
    return ("title", f"{title}:{record.get('year') or ''}")


def grouped_citations(source_records: dict[str, list[dict[str, Any]]]) -> tuple[dict[str, int], dict[str, int], int]:
    """Group identical citing works across sources using exact evidence only."""
    ordered_sources = list(source_records)
    records: list[tuple[str, dict[str, Any]]] = [
        (source, record) for source in ordered_sources for record in source_records[source]
    ]
    parents = list(range(len(records)))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left, right = root(left), root(right)
        if left != right:
            parents[right] = left

    owners: dict[tuple[str, str], int] = {}
    for index, (_, record) in enumerate(records):
        for identity in record_identities(record):
            if identity in owners:
                union(index, owners[identity])
            else:
                owners[identity] = index

    # Exact title/year is only allowed when a component has no conflicting
    # DOI or arXiv identifiers. No fuzzy title matching is used.
    components: dict[int, list[int]] = {}
    for index in range(len(records)):
        components.setdefault(root(index), []).append(index)
    title_owners: dict[tuple[str, str], int] = {}
    for component in list(components.values()):
        strong = set().union(*(record_identities(records[i][1]) for i in component))
        conflicting = sum(kind == "doi" for kind, _ in strong) > 1 or sum(kind == "arxiv" for kind, _ in strong) > 1
        if conflicting:
            continue
        for index in component:
            title_key = record_title_key(records[index][1])
            if title_key and title_key in title_owners:
                union(index, title_owners[title_key])
            elif title_key:
                title_owners[title_key] = index

    final: dict[int, set[str]] = {}
    for index, (source, _) in enumerate(records):
        final.setdefault(root(index), set()).add(source)
    totals = {source: 0 for source in ordered_sources}
    extras = {source: 0 for source in ordered_sources}
    for sources in final.values():
        for source in sources:
            totals[source] += 1
        for index, source in enumerate(ordered_sources):
            if source in sources and not any(item in sources for item in ordered_sources[:index]):
                extras[source] += 1
                break
    return totals, extras, len(final)


def openalex_citations(publication: dict[str, str]) -> list[dict[str, Any]]:
    candidates = get_json(
        "https://api.openalex.org/works",
        {"search": publication["title"], "per-page": "10", "mailto": CONTACT},
    ).get("results", [])
    target = normalize_title(publication["title"])
    work = next((item for item in candidates if normalize_title(item.get("title")) == target), None)
    if work is None and candidates:
        scored = sorted(candidates, key=lambda item: difflib.SequenceMatcher(None, target, normalize_title(item.get("title"))).ratio(), reverse=True)
        best = scored[0]
        if difflib.SequenceMatcher(None, target, normalize_title(best.get("title"))).ratio() >= 0.82:
            work = best
    if work is None:
        raise RuntimeError(f"OpenAlex could not resolve publication: {publication['title']}")
    cited_by = work.get("cited_by_api_url")
    if not cited_by:
        return []
    records: list[dict[str, Any]] = []
    cursor = "*"
    for _ in range(50):
        page = get_json(cited_by, {"per-page": "200", "cursor": cursor, "mailto": CONTACT})
        records.extend(page.get("results", []))
        cursor = (page.get("meta") or {}).get("next_cursor")
        if not cursor or not page.get("results"):
            break
        time.sleep(0.1)
    return [
        canonical_record(item.get("title"), item.get("publication_year"), item.get("primary_location", {}).get("landing_page_url"), (item.get("ids") or {}).get("doi"), arxiv_from_identifiers((item.get("ids") or {}).get("arxiv")), source_id=item.get("id"))
        for item in records
    ]


def serpapi_citations(publication: dict[str, str]) -> tuple[list[dict[str, Any]], list[str]]:
    if not SERPAPI_KEY:
        raise RuntimeError("SERPAPI_API_KEY is not configured")
    search = get_json("https://serpapi.com/search.json", {"engine": "google_scholar", "q": publication["title"], "hl": "en", "num": "20", "api_key": SERPAPI_KEY})
    target = normalize_title(publication["title"])
    cites_ids: list[str] = []
    for result in search.get("organic_results", []):
        if normalize_title(result.get("title")) != target and publication["arxiv"] not in (result.get("link") or ""):
            continue
        cited_by = ((result.get("inline_links") or {}).get("cited_by") or {}).get("link") or ""
        cites_ids.extend(parse_qs(urlparse(cited_by).query).get("cites", []))
    cites_ids = list(dict.fromkeys(cites_ids))
    if not cites_ids:
        raise RuntimeError(f"Google Scholar result has no citation ID for {publication['title']}")
    records: list[dict[str, Any]] = []
    for cites in cites_ids:
        for page_number in range(50):
            page = get_json("https://serpapi.com/search.json", {"engine": "google_scholar", "cites": cites, "as_sdt": "0", "hl": "en", "num": "20", "start": str(page_number * 20), "api_key": SERPAPI_KEY})
            batch = page.get("organic_results", [])
            for result in batch:
                summary = (result.get("publication_info") or {}).get("summary")
                records.append(canonical_record(result.get("title"), summary, result.get("link"), source_id=result.get("result_id")))
            if len(batch) < 20 or not batch:
                break
    return records, cites_ids


def semanticscholar_citations(arxiv: str) -> list[dict[str, Any]]:
    headers = {"Accept": "application/json"}
    if S2_KEY:
        headers["x-api-key"] = S2_KEY
    paper = get_json(f"https://api.semanticscholar.org/graph/v1/paper/ARXIV:{quote(arxiv)}", {"fields": "paperId,title,year,externalIds,url"}, headers=headers, retries=2)
    paper_id = paper.get("paperId")
    if not paper_id:
        return []
    records: list[dict[str, Any]] = []
    for page_number in range(100):
        page = get_json(f"https://api.semanticscholar.org/graph/v1/paper/{quote(paper_id, safe='')}/citations", {"fields": "paperId,title,year,externalIds,url", "limit": "100", "offset": str(page_number * 100)}, headers=headers, retries=2)
        batch = [entry.get("citingPaper") or {} for entry in page.get("data", [])]
        for item in batch:
            ids = item.get("externalIds") or {}
            records.append(canonical_record(item.get("title"), item.get("year"), item.get("url"), ids.get("DOI"), ids.get("ArXiv"), ids.get("PubMed"), item.get("paperId")))
        if len(batch) < 100:
            break
        time.sleep(0.2)
    return records


def ads_citations(publication: dict[str, str]) -> tuple[list[dict[str, Any]], str]:
    if not ADS_TOKEN:
        raise RuntimeError("ADS_API_TOKEN is not configured")
    headers = {"Authorization": f"Bearer {ADS_TOKEN}"}
    fields = "bibcode,title,doi,identifier,year"
    search = get_json("https://api.adsabs.harvard.edu/v1/search/query", {"q": f"identifier:arXiv:{publication['arxiv']}", "fl": fields, "rows": "5"}, headers=headers)
    docs = ((search.get("response") or {}).get("docs") or [])
    if not docs:
        search = get_json("https://api.adsabs.harvard.edu/v1/search/query", {"q": f'title:"{publication["title"]}"', "fl": fields, "rows": "5"}, headers=headers)
        docs = ((search.get("response") or {}).get("docs") or [])
    if not docs or not docs[0].get("bibcode"):
        raise RuntimeError(f"ADS could not resolve publication: {publication['title']}")
    bibcode = docs[0]["bibcode"]
    cited = get_json("https://api.adsabs.harvard.edu/v1/search/query", {"q": f"citations({bibcode})", "fl": fields, "rows": "2000"}, headers=headers)
    records: list[dict[str, Any]] = []
    for item in ((cited.get("response") or {}).get("docs") or []):
        identifiers = item.get("identifier") or []
        title = item.get("title", "")
        title = title[0] if isinstance(title, list) and title else title
        doi = item.get("doi")
        doi = doi[0] if isinstance(doi, list) and doi else doi
        records.append(canonical_record(title, item.get("year"), source_id=item.get("bibcode"), doi=doi, arxiv=arxiv_from_identifiers(identifiers)))
    return records, bibcode


def collect_for_publication(publication: dict[str, str]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any], list[str]]:
    sources: dict[str, list[dict[str, Any]]] = {}
    metadata: dict[str, Any] = {}
    warnings: list[str] = []
    if SERPAPI_KEY:
        try:
            sources["google_scholar"], metadata["google_scholar_ids"] = serpapi_citations(publication)
        except Exception as exc:
            warnings.append(f"Google Scholar {publication['arxiv']}: {exc}")
    if "google_scholar" not in sources:
        try:
            sources["openalex"] = openalex_citations(publication)
        except Exception as exc:
            warnings.append(f"OpenAlex {publication['arxiv']}: {exc}")
    try:
        sources["semantic_scholar"] = semanticscholar_citations(publication["arxiv"])
    except Exception as exc:
        warnings.append(f"Semantic Scholar {publication['arxiv']}: {exc}")
    if ADS_TOKEN:
        try:
            sources["ads"], metadata["ads_bibcode"] = ads_citations(publication)
        except Exception as exc:
            warnings.append(f"ADS {publication['arxiv']}: {exc}")
    return sources, metadata, warnings


def write_manual_baseline() -> int:
    """Keep a verified Google Scholar total until citing-work access is configured."""
    previous: dict[str, Any] = {}
    try:
        previous = json.loads(OUTPUT.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    output = {
        "total_citations": MANUAL_SCHOLAR_BASELINE,
        "last_updated": dt.datetime.now(dt.timezone.utc).date().isoformat(),
        "status": "ok",
        "sources": ["google_scholar"],
        "source_totals": {"google_scholar": MANUAL_SCHOLAR_BASELINE},
        "source_extras": {"google_scholar": MANUAL_SCHOLAR_BASELINE},
        "deduplication": "Verified Google Scholar baseline. Add SERPAPI_API_KEY to enumerate citing works and reconcile Semantic Scholar/ADS additions without double counting.",
        "works": previous.get("works", []),
        "warnings": ["Google Scholar baseline is manually supplied as 168 until SERPAPI_API_KEY is configured."],
    }
    OUTPUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote verified Google Scholar baseline of {MANUAL_SCHOLAR_BASELINE} citations to {OUTPUT}")
    return 0


def main() -> int:
    if not SERPAPI_KEY:
        return write_manual_baseline()

    total_citations = 0
    aggregate_totals: dict[str, int] = {}
    aggregate_extras: dict[str, int] = {}
    work_summaries: list[dict[str, Any]] = []
    warnings: list[str] = []
    successful_sources = 0

    for publication in PUBLICATIONS:
        source_records, metadata, paper_warnings = collect_for_publication(publication)
        warnings.extend(paper_warnings)
        successful_sources += len(source_records)
        totals, extras, paper_total = grouped_citations(source_records)
        total_citations += paper_total
        for source, value in totals.items():
            aggregate_totals[source] = aggregate_totals.get(source, 0) + value
        for source, value in extras.items():
            aggregate_extras[source] = aggregate_extras.get(source, 0) + value
        work_summaries.append({"title": publication["title"], "arxiv": publication["arxiv"], "deduplicated_citations": paper_total, "source_totals": totals, "source_extras": extras, **metadata})

    if successful_sources == 0:
        print("No citation index responded; preserving the previous result.", file=sys.stderr)
        return 0

    output: dict[str, Any] = {
        "total_citations": total_citations,
        "last_updated": dt.datetime.now(dt.timezone.utc).date().isoformat(),
        "status": "ok",
        "sources": list(aggregate_totals),
        "source_totals": aggregate_totals,
        "source_extras": aggregate_extras,
        "deduplication": "Google Scholar baseline; Semantic Scholar and ADS add only exact-ID or exact normalized-title/year matches not already present. OpenAlex is the fallback baseline when Google Scholar credentials are unavailable. No fuzzy matching.",
        "works": work_summaries,
    }
    if warnings:
        output["warnings"] = warnings
    OUTPUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote {total_citations} deduplicated citations to {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
