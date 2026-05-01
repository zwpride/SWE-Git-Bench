"""Parse git conflict markers from file content."""

from dataclasses import dataclass

# Number of context lines to include before/after a conflict block
CONTEXT_LINES = 10


@dataclass
class ConflictBlock:
    ours: str
    theirs: str
    start_line: int
    end_line: int
    context_before: str
    context_after: str
    ground_truth: str = ""  # filled later by comparing with GT


def parse_conflict_blocks(
    file_content: str,
    context_lines: int = CONTEXT_LINES,
) -> list[ConflictBlock]:
    """Parse conflict markers and extract ConflictBlock instances.

    Matches standard git conflict markers:
        <<<<<<< ...
        (ours content)
        =======
        (theirs content)
        >>>>>>> ...
    """
    lines = file_content.splitlines(keepends=True)
    blocks: list[ConflictBlock] = []

    i = 0
    while i < len(lines):
        if lines[i].startswith("<<<<<<<"):
            start_line = i  # 0-indexed
            ours_lines: list[str] = []
            theirs_lines: list[str] = []
            in_ours = True
            j = i + 1

            while j < len(lines):
                if lines[j].startswith("======="):
                    in_ours = False
                    j += 1
                    continue
                if lines[j].startswith(">>>>>>>"):
                    end_line = j  # 0-indexed, inclusive
                    # Context before
                    ctx_start = max(0, start_line - context_lines)
                    ctx_before = "".join(lines[ctx_start:start_line])
                    # Context after
                    ctx_end = min(len(lines), end_line + 1 + context_lines)
                    ctx_after = "".join(lines[end_line + 1:ctx_end])

                    blocks.append(ConflictBlock(
                        ours="".join(ours_lines),
                        theirs="".join(theirs_lines),
                        start_line=start_line,
                        end_line=end_line,
                        context_before=ctx_before,
                        context_after=ctx_after,
                    ))
                    i = j + 1
                    break
                if in_ours:
                    ours_lines.append(lines[j])
                else:
                    theirs_lines.append(lines[j])
                j += 1
            else:
                # Unterminated conflict block, skip
                i += 1
                continue
            continue
        i += 1

    return blocks


def fill_block_ground_truths(
    blocks: list[ConflictBlock],
    conflicted_content: str,
    ground_truth_content: str,
):
    """Align conflict blocks with ground truth to fill each block's GT.

    For single-block files: use exact prefix/suffix matching.
    The non-conflict lines before the block are an exact prefix of GT,
    and non-conflict lines after the block are an exact suffix.
    Block GT is everything in between.

    For multi-block files: use inter-block non-conflict text as separators,
    finding them in the GT to segment it into per-block GTs.
    """
    if not blocks:
        return

    conf_lines = conflicted_content.splitlines(keepends=True)
    gt_lines = ground_truth_content.splitlines(keepends=True)

    if len(blocks) == 1:
        _fill_single_block(blocks[0], conf_lines, gt_lines)
    else:
        _fill_multi_block(blocks, conf_lines, gt_lines)


def _fill_single_block(
    block: ConflictBlock,
    conf_lines: list[str],
    gt_lines: list[str],
):
    """Fill GT for a file with exactly one conflict block.

    Uses prefix/suffix matching. The prefix (before <<<<<<<) is always exact.
    The suffix is matched from the end of both files. Due to blank-line
    boundary ambiguity in git conflict markers, this may include/exclude
    1-2 blank lines at the block boundary. This is acceptable since the
    whole-file GT (stored separately) is authoritative for all main metrics.
    """
    prefix_len = block.start_line

    # Match suffix from end of both files
    after_start = block.end_line + 1
    suffix_len = 0
    max_suffix = min(len(conf_lines) - after_start, len(gt_lines) - prefix_len)
    for i in range(1, max_suffix + 1):
        if conf_lines[-i] == gt_lines[-i]:
            suffix_len = i
        else:
            break

    gt_end = len(gt_lines) - suffix_len if suffix_len > 0 else len(gt_lines)
    block.ground_truth = "".join(gt_lines[prefix_len:gt_end])


def _fill_multi_block(
    blocks: list[ConflictBlock],
    conf_lines: list[str],
    gt_lines: list[str],
):
    """Fill GT for a file with multiple conflict blocks."""
    # Strategy: find the inter-block non-conflict segments as anchors in GT.
    #
    # Build list of segments: [prefix, block0, inter01, block1, inter12, ..., suffix]
    # The inter-block segments (non-conflict text between blocks) should appear
    # in the GT in order. We find them to determine each block's GT boundaries.

    # First, handle prefix exactly (same as single-block)
    prefix_len = blocks[0].start_line

    # Handle suffix: match from end
    after_last_block = conf_lines[blocks[-1].end_line + 1:]
    suffix_len = 0
    max_suffix = min(len(after_last_block), len(gt_lines) - prefix_len)
    for i in range(1, max_suffix + 1):
        if conf_lines[-i] == gt_lines[-i]:
            suffix_len = i
        else:
            break

    gt_start = prefix_len
    gt_end_bound = len(gt_lines) - suffix_len if suffix_len > 0 else len(gt_lines)

    # Now find inter-block separators in the GT region [gt_start:gt_end_bound]
    # For each pair of consecutive blocks, the non-conflict text between them
    # must appear somewhere in the GT.

    gt_pos = gt_start

    for bi, block in enumerate(blocks):
        if bi < len(blocks) - 1:
            # Find the inter-block text (between this block's end and next block's start)
            inter_start = block.end_line + 1
            inter_end = blocks[bi + 1].start_line
            inter_lines = conf_lines[inter_start:inter_end]

            if not inter_lines:
                # Blocks are adjacent - no separator
                continue

            # Find first distinctive (non-blank) line in inter-block text
            anchor = None
            anchor_offset = 0
            for ai, line in enumerate(inter_lines):
                if line.strip():
                    anchor = line
                    anchor_offset = ai
                    break

            if anchor is None:
                # All blank lines between blocks - can't anchor reliably
                # Split remaining GT space evenly (rough approximation)
                continue

            # Search for anchor in GT from gt_pos
            found = False
            for k in range(gt_pos, gt_end_bound):
                if gt_lines[k] == anchor:
                    # Verify with more lines
                    verify_n = min(3, len(inter_lines) - anchor_offset)
                    match = True
                    for v in range(verify_n):
                        if k + v >= gt_end_bound or gt_lines[k + v] != inter_lines[anchor_offset + v]:
                            match = False
                            break
                    if match:
                        # Block GT ends somewhere before this anchor
                        # The anchor is at gt_lines[k], and there are anchor_offset
                        # blank lines before it in the inter-block text.
                        # Block GT = gt_lines[gt_pos : k - anchor_offset] roughly,
                        # but we need to be careful with blank lines.
                        # Use the anchor position minus the number of inter-block
                        # lines before the anchor as the split point.
                        block_gt_end = k - anchor_offset
                        block_gt_end = max(block_gt_end, gt_pos)
                        block.ground_truth = "".join(gt_lines[gt_pos:block_gt_end])

                        # Advance gt_pos past the inter-block text
                        inter_end_in_gt = k + (len(inter_lines) - anchor_offset)
                        gt_pos = min(inter_end_in_gt, gt_end_bound)
                        found = True
                        break

            if not found:
                # Fallback: can't find separator
                block.ground_truth = ""
        else:
            # Last block: GT is everything from gt_pos to gt_end_bound
            block.ground_truth = "".join(gt_lines[gt_pos:gt_end_bound])
