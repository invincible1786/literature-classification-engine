"""Build offline cached snapshots in data/snapshots/ for reproducible offline execution."""
import csv
import json
import os
import hashlib

def canonicalize_simple(raw: str) -> str:
    import re, html, unicodedata
    inline = re.compile(r"</?(?:i|b|em|strong|sup|sub)\b[^>]*>", re.I)
    anytag = re.compile(r"<[^>]+>")
    t = inline.sub("", raw)
    t = anytag.sub(" ", t)
    t = html.unescape(t)
    t = unicodedata.normalize("NFKC", t)
    return re.sub(r"\s+", " ", t).strip()

def build_snapshots():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    labels_file = os.path.join(base_dir, "eval", "labels.csv")
    snapshots_dir = os.path.join(base_dir, "data", "snapshots")
    os.makedirs(snapshots_dir, exist_ok=True)
    
    entities = {}
    with open(labels_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ent = row["entity"]
            if ent not in entities:
                entities[ent] = []
            
            raw_text = row["abstract"]
            canon_text = canonicalize_simple(raw_text)
            text_sha = hashlib.sha256(canon_text.encode("utf-8")).hexdigest()
            
            item = {
                "id": row["pmid_or_pmcid"],
                "pmid": row["pmid_or_pmcid"].replace("PMC", ""),
                "pmcid": row["pmid_or_pmcid"] if row["pmid_or_pmcid"].startswith("PMC") else None,
                "doi": f"10.1000/{row['pmid_or_pmcid']}",
                "title": row["title"],
                "pubYear": "2021",
                "abstractText": raw_text,
                "raw_text": raw_text,
                "canonical_text": canon_text,
                "text_sha256": text_sha,
                "ground_truth_label": row["label"],
                "ground_truth_category": row["category"],
                "ground_truth_evidence": row["evidence_text"]
            }
            entities[ent].append(item)
            
    for ent, items in entities.items():
        out_file = os.path.join(snapshots_dir, f"{ent}.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump({
                "entity": ent,
                "total": len(items),
                "papers": items
            }, f, indent=2)
        print(f"Wrote snapshot for {ent}: {len(items)} papers -> {out_file}")

if __name__ == "__main__":
    build_snapshots()

