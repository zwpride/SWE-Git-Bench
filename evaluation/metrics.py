"""Evaluation metrics for merge conflict resolution."""

import difflib
from dataclasses import dataclass, asdict


def _normalize(text: str) -> str:
    """Normalize text for comparison: strip trailing whitespace per line, ensure trailing newline."""
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(lines) + "\n" if lines else ""


def exact_match(prediction: str, ground_truth: str) -> bool:
    """Check if prediction exactly matches ground truth (after normalization)."""
    return _normalize(prediction) == _normalize(ground_truth)


def normalized_edit_similarity(prediction: str, ground_truth: str) -> float:
    """Compute 1 - (edit_distance / max_len).

    Uses character-level Levenshtein distance approximated via difflib.
    Returns a float in [0, 1], where 1.0 means identical.
    """
    p = _normalize(prediction)
    g = _normalize(ground_truth)
    if not p and not g:
        return 1.0
    # Use SequenceMatcher ratio at character level as proxy
    return difflib.SequenceMatcher(None, p, g).ratio()


def line_diff_similarity(prediction: str, ground_truth: str) -> float:
    """Compute line-level similarity using difflib.SequenceMatcher.

    Returns a float in [0, 1], where 1.0 means identical.
    """
    p_lines = _normalize(prediction).splitlines()
    g_lines = _normalize(ground_truth).splitlines()
    if not p_lines and not g_lines:
        return 1.0
    return difflib.SequenceMatcher(None, p_lines, g_lines).ratio()


def _normalize_block(text: str) -> str:
    """Normalize a conflict block for comparison.

    Strips leading/trailing blank lines (boundary ambiguity from git markers)
    and trailing whitespace per line.
    """
    lines = text.splitlines()
    # Strip leading blank lines
    while lines and not lines[0].strip():
        lines = lines[1:]
    # Strip trailing blank lines
    while lines and not lines[-1].strip():
        lines = lines[:-1]
    return "\n".join(line.rstrip() for line in lines) + "\n" if lines else ""


def block_level_accuracy(
    pred_blocks: list[str],
    gt_blocks: list[str],
) -> float:
    """Compute per-block exact match accuracy.

    Each block is compared after stripping leading/trailing blank lines
    (to handle boundary ambiguity from git conflict markers) and
    normalizing trailing whitespace.
    Returns fraction of blocks that match exactly.
    """
    if not gt_blocks:
        return 1.0 if not pred_blocks else 0.0
    n = min(len(pred_blocks), len(gt_blocks))
    matches = sum(
        1 for p, g in zip(pred_blocks[:n], gt_blocks[:n])
        if _normalize_block(p) == _normalize_block(g)
    )
    return matches / len(gt_blocks)


@dataclass
class EvalResult:
    model_name: str
    conflict_id: str          # repo__merge_commit_sha
    instance_id: str           # SWE-bench instance id
    prompt_format: str         # "whole_file" | "conflict_block"
    file_path: str
    exact_match: bool
    edit_similarity: float
    line_similarity: float
    block_accuracy: float      # only meaningful for conflict_block format
    raw_prediction: str
    raw_response: str

    def to_dict(self) -> dict:
        return asdict(self)


def compute_file_metrics(
    prediction: str,
    ground_truth: str,
    pred_blocks: list[str] | None = None,
    gt_blocks: list[str] | None = None,
) -> dict:
    """Compute all metrics for a single file prediction."""
    em = exact_match(prediction, ground_truth)
    edit_sim = normalized_edit_similarity(prediction, ground_truth)
    line_sim = line_diff_similarity(prediction, ground_truth)
    block_acc = 0.0
    if pred_blocks is not None and gt_blocks is not None:
        block_acc = block_level_accuracy(pred_blocks, gt_blocks)
    return {
        "exact_match": em,
        "edit_similarity": round(edit_sim, 4),
        "line_similarity": round(line_sim, 4),
        "block_accuracy": round(block_acc, 4),
    }
