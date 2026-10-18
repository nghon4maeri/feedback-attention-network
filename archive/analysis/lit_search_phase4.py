"""Phase 4 literature search: fallback losses + gating scope + novelty check.

OpenAlex + Semantic Scholar. Queries targeted at:
  - negative-area / false-positive-penalty Dice loss
  - Lovasz hinge / soft-IoU
  - background-aware loss segmentation
  - encoder-only vs decoder-only attention gating
  - novelty: background-aware feedback loop segmentation
"""
import json
import time
import urllib.parse
import urllib.request

OPENALEX = "https://api.openalex.org/works"
S2 = "https://api.semanticscholar.org/graph/v1/paper/search"

QUERIES = [
    ("neg_area_dice", "negative area dice loss segmentation"),
    ("fp_penalty_loss", "false positive penalty loss medical segmentation"),
    ("lovasz_hinge", "lovasz hinge soft IoU loss segmentation"),
    ("bg_aware_loss", "background aware loss segmentation"),
    ("gating_scope", "decoder attention gating segmentation encoder"),
    ("bg_feedback_loop", "background aware feedback loop segmentation"),
]


def search_openalex(q):
    url = OPENALEX + "?" + urllib.parse.urlencode(
        {"search": q, "per-page": "6",
         "select": "title,publication_year,doi,cited_by_count"})
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r).get("results", [])


def search_s2(q):
    url = S2 + "?" + urllib.parse.urlencode(
        {"query": q, "limit": "6", "fields": "title,year,externalIds,citationCount"})
    req = urllib.request.Request(url, headers={"User-Agent": "research-engineer"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r).get("data", [])


for key, q in QUERIES:
    print(f"\n========== {key} ==========")
    try:
        for w in search_openalex(q):
            title = (w.get("title") or "")[:95]
            print(f"  OA  {w.get('publication_year')} | cited:{w.get('cited_by_count',0):>4} | {title}")
            if w.get("doi"):
                print(f"        {w['doi']}")
    except Exception as e:
        print(f"  OpenAlex error: {e}")
    time.sleep(1)
    try:
        for w in search_s2(q):
            title = (w.get("title") or "")[:95]
            ext = w.get("externalIds") or {}
            print(f"  S2  {w.get('year')} | cited:{w.get('citationCount',0):>4} | {title}")
            if ext.get("DOI"):
                print(f"        DOI:{ext['DOI']}")
    except Exception as e:
        print(f"  SemanticScholar error: {e}")
    time.sleep(2)
