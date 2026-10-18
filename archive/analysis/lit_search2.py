"""Round 2: targeted literature search."""
import json
import urllib.parse
import urllib.request

QUERIES = {
    "prawnet": "PraNet parallel reverse attention polyp",
    "uacanet": "UACANet uncertainty augmented context attention polyp",
    "prior_mask_guidance": "prior mask guidance iterative refinement medical segmentation",
    "negative_mining_seg": "hard negative mining background context segmentation",
    "soft_gate_binarization": "soft gating binarized attention gradient segmentation network",
    "camouflaged_iterative": "iterative feedback camouflage segmentation background suppression",
}

for key, q in QUERIES.items():
    url = ("https://api.openalex.org/works?"
           + urllib.parse.urlencode({
               "search": q, "per-page": "6",
               "select": "title,publication_year,doi,cited_by_count",
           }))
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            data = json.load(r)
    except Exception as e:
        print(f"\n=== {key} === ERROR {e}")
        continue
    print(f"\n=== {key} ===")
    for w in data.get("results", []):
        title = (w.get("title") or "")[:95]
        year = w.get("publication_year")
        doi = w.get("doi")
        cited = w.get("cited_by_count", 0)
        print(f"  {year} | cited:{cited:>4} | {title}")
        if doi:
            print(f"         {doi}")
