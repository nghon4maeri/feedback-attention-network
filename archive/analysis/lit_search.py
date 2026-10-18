"""Literature search for FANet Phase 2+3 grounding via OpenAlex."""
import json
import urllib.parse
import urllib.request

QUERIES = {
    "background_aware_feedback": "background aware feedback segmentation",
    "dual_path_polyp": "dual path feedback polyp segmentation",
    "reverse_attention": "reverse attention iterative segmentation",
    "straight_through_estimator": "straight-through estimator binary attention network",
    "over_segmentation_loss": "over-segmentation false positive penalty boundary loss segmentation",
    "feedback_attention_network": "feedback attention network biomedical segmentation",
}

for key, q in QUERIES.items():
    url = ("https://api.openalex.org/works?"
           + urllib.parse.urlencode({
               "search": q, "per-page": "8",
               "select": "title,publication_year,doi,cited_by_count",
           }))
    with urllib.request.urlopen(url, timeout=30) as r:
        data = json.load(r)
    print(f"\n=== {key} ===")
    for w in data.get("results", []):
        title = (w.get("title") or "")[:95]
        year = w.get("publication_year")
        doi = w.get("doi")
        cited = w.get("cited_by_count", 0)
        print(f"  {year} | cited:{cited:>4} | {title}")
        if doi:
            print(f"         {doi}")
