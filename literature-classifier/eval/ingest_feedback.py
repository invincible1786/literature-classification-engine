"""Curator feedback ingestion script: validates and stages feedback into candidate review queue."""
import os
import sys
import json
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def ingest_feedback(feedback_path: str, candidate_output_path: str) -> Tuple[int, int]:
    if not os.path.exists(feedback_path):
        print(f"No feedback file found at {feedback_path}")
        return 0, 0

    valid_entries: List[Dict[str, Any]] = []
    skipped = 0

    with open(feedback_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except Exception as e:
                logger.warning(f"Line {line_no}: JSON parse error: {e}")
                skipped += 1
                continue

            # Validate schema
            paper_id = record.get("paper_id")
            entity = record.get("entity")
            criterion = record.get("criterion")
            suggested_label = record.get("suggested_label")

            if not (paper_id and entity and criterion and suggested_label):
                skipped += 1
                continue

            if suggested_label not in ("qualifies", "does_not_qualify", "insufficient_evidence"):
                skipped += 1
                continue

            valid_entries.append({
                "paper_id": str(paper_id),
                "entity": str(entity),
                "criterion": str(criterion),
                "suggested_label": str(suggested_label),
                "timestamp": record.get("timestamp"),
                "model_revision": record.get("model_revision"),
                "notes": record.get("notes"),
                "curator_status": "pending_review"
            })

    # Write staged candidates
    with open(candidate_output_path, "w", encoding="utf-8") as out_f:
        json.dump(valid_entries, out_f, indent=2)

    return len(valid_entries), skipped


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    feedback_file = os.path.join(base_dir, "feedback.jsonl")
    candidate_file = os.path.join(base_dir, "eval", "candidate_feedback.json")

    print(f"Reading curator queue from {feedback_file}...")
    valid, skipped = ingest_feedback(feedback_file, candidate_file)
    print(f"Processed {valid} valid candidate entries for curator review ({skipped} skipped/malformed).")
    print(f"Staged in {candidate_file}")


if __name__ == "__main__":
    from typing import Tuple
    main()
