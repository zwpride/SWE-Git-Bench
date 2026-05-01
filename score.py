#!/usr/bin/env python3
"""Score SWE-Git-Bench prediction JSONL files.

Prediction rows must include:
  dataset, model_name, conflict_id, file_path, raw_prediction

The eval dataset JSONL files are the Dataverse `eval_dataset_*.jsonl` files.
They contain the ground-truth resolution for each conflicted file.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from evaluation.metrics import compute_file_metrics


def load_ground_truth(eval_files: list[Path]) -> dict[tuple[str, str, str], str]:
    ground_truth: dict[tuple[str, str, str], str] = {}
    for eval_file in eval_files:
        dataset = eval_file.stem.removeprefix("eval_dataset_")
        with eval_file.open() as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                for file_record in record.get("files", []):
                    key = (dataset, record["conflict_id"], file_record["path"])
                    ground_truth[key] = file_record["ground_truth"]
    return ground_truth


def infer_dataset(path: Path) -> str | None:
    name = path.name
    for dataset in ("lite", "Verified", "Multilingual", "Multimodal"):
        if name.startswith(f"{dataset}_"):
            return dataset
    return None


def iter_predictions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dataset", nargs="+", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args()

    ground_truth = load_ground_truth(args.eval_dataset)
    rows = []
    missing = 0

    inferred_dataset = infer_dataset(args.predictions)
    for pred in iter_predictions(args.predictions):
        dataset = pred.get("dataset") or inferred_dataset
        if dataset is None:
            raise SystemExit(
                "Prediction rows must include `dataset`, or the prediction "
                "filename must start with lite_, Verified_, Multilingual_, or Multimodal_."
            )
        key = (dataset, pred["conflict_id"], pred["file_path"])
        expected = ground_truth.get(key)
        if expected is None:
            missing += 1
            continue
        metrics = compute_file_metrics(pred.get("raw_prediction", ""), expected)
        rows.append(
            {
                "dataset": dataset,
                "model_name": pred["model_name"],
                "conflict_id": pred["conflict_id"],
                "file_path": pred["file_path"],
                **metrics,
            }
        )

    if not rows:
        raise SystemExit("No predictions matched the provided eval datasets.")

    em = sum(1 for row in rows if row["exact_match"]) / len(rows)
    es = sum(float(row["edit_similarity"]) for row in rows) / len(rows)
    print(f"matched={len(rows)} missing={missing} EM={em * 100:.2f} ES={es * 100:.2f}")

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
