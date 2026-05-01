"""Inter-rater reliability (IRR) analysis for the SWE-Git-Bench failure taxonomy.

Computes Cohen's kappa and related statistics between two judge files that were
produced by analysis/failure_taxonomy.py.  Both files are matched on `sample_id`
(the 16-character SHA-1 key computed from dataset/prompt_format/model_name/
conflict_id/file_path).

Usage
-----
python3 analysis/irr_kappa.py \\
    --judge1-file results/failure_taxonomy_deepseek_v32.jsonl \\
    --judge2-file results/failure_taxonomy_gemma3_27b.jsonl

The released analysis archive already includes both judge files used in the
paper. New judge runs can be generated with any OpenAI-compatible endpoint by
using the same prompt format described in the paper appendix.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

CATEGORIES = [
    "pick-wrong-side",
    "truncated",
    "hallucinated",
    "format-error",
    "equivalent-variant",
]

# model_name values that belong to DeepSeek-V3.2 (the judge-1 model family).
# We use a substring match so "DeepSeek-V3.2-Speciale" is also captured.
JUDGE1_MODEL_SUBSTRING = "DeepSeek-V3.2"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    print(f"[warn] skipping malformed line in {path}: {exc}")
    return records


def index_by_sample_id(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for rec in records:
        sid = rec.get("sample_id")
        if sid:
            if sid in index:
                print(f"[warn] duplicate sample_id {sid!r} – keeping last occurrence")
            index[sid] = rec
    return index


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def cohen_kappa(
    labels1: list[str],
    labels2: list[str],
    categories: list[str],
) -> float:
    """Compute Cohen's kappa for two equal-length label sequences.

    Parameters
    ----------
    labels1, labels2 : list of str
        Parallel label sequences from rater 1 and rater 2.
    categories : list of str
        The full ordered category list.

    Returns
    -------
    float
        Cohen's kappa, or NaN when the denominator is zero.
    """
    n = len(labels1)
    if n == 0:
        return float("nan")

    # Observed agreement
    po = sum(a == b for a, b in zip(labels1, labels2)) / n

    # Expected agreement
    counter1: Counter[str] = Counter(labels1)
    counter2: Counter[str] = Counter(labels2)
    pe = sum((counter1[c] / n) * (counter2[c] / n) for c in categories)

    if abs(1.0 - pe) < 1e-12:
        return float("nan")

    return (po - pe) / (1.0 - pe)


def confusion_matrix(
    labels1: list[str],
    labels2: list[str],
    categories: list[str],
) -> dict[str, dict[str, int]]:
    """Build a confusion matrix indexed by [true (judge1)][predicted (judge2)]."""
    matrix: dict[str, dict[str, int]] = {c: {d: 0 for d in categories} for c in categories}
    for a, b in zip(labels1, labels2):
        if a in matrix and b in matrix[a]:
            matrix[a][b] += 1
        # Labels outside the canonical set are silently ignored (e.g. judge-error).
    return matrix


def per_category_stats(
    matrix: dict[str, dict[str, int]],
    categories: list[str],
) -> dict[str, dict[str, float]]:
    """Compute per-category precision / recall / F1 from judge1's perspective.

    - Precision (from judge1's perspective): of all samples judge2 labels as C,
      how many did judge1 also label as C?  (judge1 is treated as ground truth.)
    - Recall (from judge1's perspective): of all samples judge1 labels as C,
      how many did judge2 also label as C?
    """
    stats: dict[str, dict[str, float]] = {}
    for c in categories:
        tp = matrix[c][c]
        # judge1 says C: sum of row c
        row_sum = sum(matrix[c].values())
        # judge2 says C: sum of column c
        col_sum = sum(matrix[r][c] for r in categories)

        recall = tp / row_sum if row_sum else float("nan")
        precision = tp / col_sum if col_sum else float("nan")
        if (not isinstance(precision, float) or not isinstance(recall, float)
                or (precision + recall) == 0):
            f1 = float("nan")
        else:
            f1 = 2 * precision * recall / (precision + recall)

        stats[c] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "judge1_count": row_sum,
            "judge2_count": col_sum,
            "agreement_count": tp,
        }
    return stats


def agreement_by_model_group(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    judge1_model_substring: str,
) -> dict[str, dict[str, Any]]:
    """Measure whether judge1's agreement rate differs for its own-model failures.

    'Own model' samples are those where the evaluated model_name contains
    `judge1_model_substring` (e.g. "DeepSeek-V3.2").  The rationale: a
    self-lenient judge might systematically rate its own failures differently.
    """
    own_agreed = own_total = 0
    other_agreed = other_total = 0

    for rec1, rec2 in pairs:
        model_name = str(rec1.get("model_name", ""))
        agree = rec1.get("category") == rec2.get("category")
        if judge1_model_substring in model_name:
            own_total += 1
            if agree:
                own_agreed += 1
        else:
            other_total += 1
            if agree:
                other_agreed += 1

    own_rate = own_agreed / own_total if own_total else float("nan")
    other_rate = other_agreed / other_total if other_total else float("nan")

    return {
        "own_model": {
            "substring_matched": judge1_model_substring,
            "n": own_total,
            "agreed": own_agreed,
            "agreement_rate": own_rate,
        },
        "other_models": {
            "n": other_total,
            "agreed": other_agreed,
            "agreement_rate": other_rate,
        },
        "difference": (
            own_rate - other_rate
            if not (isinstance(own_rate, float) and own_rate != own_rate)
            and not (isinstance(other_rate, float) and other_rate != other_rate)
            else float("nan")
        ),
        "note": (
            "Positive difference means judge1 agrees MORE on its own-model "
            "failures (leniency); negative means MORE on other-model failures."
        ),
    }


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def _fmt(value: float, pct: bool = False, decimals: int = 4) -> str:
    if isinstance(value, float) and value != value:  # NaN check
        return "  N/A "
    if pct:
        return f"{value * 100:6.2f}%"
    return f"{value:.{decimals}f}"


def print_judge1_profile(
    records: list[dict[str, Any]],
    judge1_model_substring: str,
) -> None:
    print("=" * 68)
    print("JUDGE-1 FILE PROFILE")
    print("=" * 68)

    unique_ids = set(r["sample_id"] for r in records if "sample_id" in r)
    print(f"  Total records        : {len(records)}")
    print(f"  Unique sample_ids    : {len(unique_ids)}")

    cat_counter: Counter[str] = Counter(r.get("category") for r in records)
    print("\n  Category distribution:")
    for cat in CATEGORIES:
        n = cat_counter.get(cat, 0)
        pct = n / len(records) * 100 if records else 0
        print(f"    {cat:<22s}: {n:4d}  ({pct:5.1f}%)")
    other_cats = {k: v for k, v in cat_counter.items() if k not in CATEGORIES}
    for cat, n in sorted(other_cats.items()):
        pct = n / len(records) * 100 if records else 0
        print(f"    {cat:<22s}: {n:4d}  ({pct:5.1f}%)  [non-taxonomy]")

    own = [r for r in records if judge1_model_substring in str(r.get("model_name", ""))]
    other = [r for r in records if judge1_model_substring not in str(r.get("model_name", ""))]
    print(f"\n  Samples from judge1's own model family ({judge1_model_substring!r}):")
    print(f"    Own-model samples   : {len(own)}")
    print(f"    Other-model samples : {len(other)}")

    model_counter: Counter[str] = Counter(r.get("model_name") for r in records)
    print("\n  Per-model breakdown:")
    for model, n in sorted(model_counter.items()):
        tag = " *" if judge1_model_substring in str(model) else ""
        print(f"    {model:<45s}: {n:3d}{tag}")
    print("  (* = belongs to judge1's own model family)")


def print_irr_summary(
    kappa: float,
    pct_agree: float,
    n_matched: int,
    n_j1: int,
    n_j2: int,
    n_only_j1: int,
    n_only_j2: int,
    per_cat: dict[str, dict[str, float]],
    bias: dict[str, Any],
) -> None:
    print()
    print("=" * 68)
    print("INTER-RATER RELIABILITY (IRR) ANALYSIS")
    print("=" * 68)

    print(f"\n  Overlap / coverage")
    print(f"    Judge-1 total samples    : {n_j1}")
    print(f"    Judge-2 total samples    : {n_j2}")
    print(f"    Matched (both judges)    : {n_matched}")
    print(f"    Only in judge-1          : {n_only_j1}")
    print(f"    Only in judge-2          : {n_only_j2}")

    print(f"\n  Overall agreement statistics (on {n_matched} matched pairs)")
    print(f"    Percent agreement        : {_fmt(pct_agree, pct=True)}")
    print(f"    Cohen's kappa            : {_fmt(kappa)}")
    interp = _interpret_kappa(kappa)
    print(f"    Kappa interpretation     : {interp}")

    print(f"\n  Per-category statistics (judge1 = reference)")
    header = f"  {'Category':<22s}  {'J1-N':>5}  {'J2-N':>5}  {'Agree':>5}  {'Recall':>7}  {'Precision':>9}  {'F1':>7}"
    print(header)
    print("  " + "-" * 66)
    for cat in CATEGORIES:
        s = per_cat[cat]
        print(
            f"  {cat:<22s}"
            f"  {s['judge1_count']:5d}"
            f"  {s['judge2_count']:5d}"
            f"  {s['agreement_count']:5d}"
            f"  {_fmt(s['recall'], pct=True):>8}"
            f"  {_fmt(s['precision'], pct=True):>9}"
            f"  {_fmt(s['f1'], pct=True):>8}"
        )

    print(f"\n  Self-leniency bias analysis")
    own = bias["own_model"]
    oth = bias["other_models"]
    diff = bias["difference"]
    print(f"    Judge-1 own-model failures ({own['substring_matched']!r})")
    print(f"      n={own['n']}, agreed={own['agreed']}, "
          f"agreement rate={_fmt(own['agreement_rate'], pct=True)}")
    print(f"    Other-model failures")
    print(f"      n={oth['n']}, agreed={oth['agreed']}, "
          f"agreement rate={_fmt(oth['agreement_rate'], pct=True)}")
    if isinstance(diff, float) and diff == diff:
        sign = "+" if diff >= 0 else ""
        print(f"    Difference (own - other) : {sign}{diff * 100:.2f} pp")
    print(f"    Note: {bias['note']}")


def _interpret_kappa(kappa: float) -> str:
    if isinstance(kappa, float) and kappa != kappa:  # NaN
        return "N/A"
    if kappa < 0:
        return "poor (< 0)"
    if kappa < 0.20:
        return "slight (0.00–0.20)"
    if kappa < 0.40:
        return "fair (0.20–0.40)"
    if kappa < 0.60:
        return "moderate (0.40–0.60)"
    if kappa < 0.80:
        return "substantial (0.60–0.80)"
    return "almost perfect (0.80–1.00)"


def print_gemma_run_command() -> None:
    print()
    print("=" * 68)
    print("JUDGE RE-RUN NOTE")
    print("=" * 68)
    print("""
  The Dataverse analysis archive includes both judge JSONL files used in the
  paper. To generate a fresh second-judge file, use the judge prompt from the
  paper appendix with any OpenAI-compatible model endpoint and keep the same
  sample_id fields so this script can join the two label sets.
""")


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> None:
    judge1_path = Path(args.judge1_file)
    judge2_path = Path(args.judge2_file)
    output_path = Path(args.output)

    # ------------------------------------------------------------------ #
    # Load judge-1 file and print its profile
    # ------------------------------------------------------------------ #
    print(f"\nLoading judge-1 file: {judge1_path}")
    j1_records = load_jsonl(judge1_path)
    j1_index = index_by_sample_id(j1_records)

    print_judge1_profile(j1_records, JUDGE1_MODEL_SUBSTRING)

    # ------------------------------------------------------------------ #
    # Load judge-2 file (may not exist yet)
    # ------------------------------------------------------------------ #
    if not judge2_path.exists():
        print(f"\n[warn] Judge-2 file does not exist yet: {judge2_path}")
        print("       Run the Gemma-3-27B command below to generate it.")
        print_gemma_run_command()
        return

    print(f"\nLoading judge-2 file: {judge2_path}")
    j2_records = load_jsonl(judge2_path)
    j2_index = index_by_sample_id(j2_records)

    n_j1 = len(j1_index)
    n_j2 = len(j2_index)

    # ------------------------------------------------------------------ #
    # Join on sample_id
    # ------------------------------------------------------------------ #
    common_ids = sorted(set(j1_index) & set(j2_index))
    only_j1 = len(set(j1_index) - set(j2_index))
    only_j2 = len(set(j2_index) - set(j1_index))

    print(f"\nMatched {len(common_ids)} sample_ids  "
          f"(only-j1: {only_j1}, only-j2: {only_j2})")

    if not common_ids:
        print("[error] No matching sample_ids found – cannot compute IRR.")
        print_gemma_run_command()
        return

    # Keep only the 5 canonical categories; skip judge-error rows.
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    skipped = 0
    for sid in common_ids:
        r1 = j1_index[sid]
        r2 = j2_index[sid]
        if r1.get("category") not in CATEGORIES or r2.get("category") not in CATEGORIES:
            skipped += 1
            continue
        pairs.append((r1, r2))

    if skipped:
        print(f"[info] Skipped {skipped} pairs with non-taxonomy categories "
              f"(e.g. judge-error)")

    labels1 = [p[0]["category"] for p in pairs]
    labels2 = [p[1]["category"] for p in pairs]

    # ------------------------------------------------------------------ #
    # Compute statistics
    # ------------------------------------------------------------------ #
    n_matched = len(pairs)
    pct_agree = sum(a == b for a, b in zip(labels1, labels2)) / n_matched
    kappa = cohen_kappa(labels1, labels2, CATEGORIES)
    matrix = confusion_matrix(labels1, labels2, CATEGORIES)
    per_cat = per_category_stats(matrix, CATEGORIES)
    bias = agreement_by_model_group(pairs, JUDGE1_MODEL_SUBSTRING)

    # ------------------------------------------------------------------ #
    # Print summary
    # ------------------------------------------------------------------ #
    print_irr_summary(
        kappa=kappa,
        pct_agree=pct_agree,
        n_matched=n_matched,
        n_j1=n_j1,
        n_j2=n_j2,
        n_only_j1=only_j1,
        n_only_j2=only_j2,
        per_cat=per_cat,
        bias=bias,
    )

    # ------------------------------------------------------------------ #
    # Save JSON output
    # ------------------------------------------------------------------ #
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "judge1_file": str(judge1_path),
        "judge2_file": str(judge2_path),
        "n_judge1": n_j1,
        "n_judge2": n_j2,
        "n_matched": n_matched,
        "n_only_judge1": only_j1,
        "n_only_judge2": only_j2,
        "n_skipped_nontaxonomy": skipped,
        "percent_agreement": pct_agree,
        "cohen_kappa": kappa,
        "kappa_interpretation": _interpret_kappa(kappa),
        "per_category": per_cat,
        "confusion_matrix": matrix,
        "self_leniency_bias": bias,
    }
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(f"\nResults saved to: {output_path}")

    print_gemma_run_command()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--judge1-file",
        type=Path,
        default=RESULTS_DIR / "failure_taxonomy_deepseek_v32.jsonl",
        help="Path to the JSONL output of the first judge (default: DeepSeek-V3.2).",
    )
    parser.add_argument(
        "--judge2-file",
        type=Path,
        default=RESULTS_DIR / "failure_taxonomy_gemma3_27b.jsonl",
        help="Path to the JSONL output of the second judge (default: Gemma-3-27B).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "irr_kappa_results.json",
        help="Where to write the IRR results JSON (default: results/irr_kappa_results.json).",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
