# SWE-Git-Bench — Anonymous Code Mirror

This directory is the anonymized source-code mirror for the
**SWE-Git-Bench** benchmark and evaluation harness, prepared for
NeurIPS 2026 Datasets & Benchmarks double-blind review.

It is the input to `../push_code.sh`, which:

1. PII-audits this tree (regex match against author names / affiliations).
2. Re-initializes a fresh git history (no upstream commit metadata).
3. Pushes to a GitHub repository owned by an anonymous account.
4. Forwards the URL to anonymous.4open.science for serving the
   review-time read-only mirror.

## Contents

```
github_swegitbench/
├── README.md
├── REPRODUCIBILITY.md
├── LICENSE
├── requirements.txt
├── score.py             # score prediction JSONL files against eval_dataset_*.jsonl
├── evaluation/          # EM / ES / LS / block-accuracy metrics
├── analysis/            # bootstrap CI and IRR scripts
├── conflict/            # conflict-block parsing helpers
├── prompts/             # WF and CB prompt templates
```

## Reproducing the main numbers

See `REPRODUCIBILITY.md` for the scoring and analysis commands. The dataset
itself (conflict files, ground-truth resolutions, and prediction matrix) is
served from the companion Harvard Dataverse anonymized preview link.

## License

CC BY 4.0 for the harness and prompt templates. Per-component licenses
inherit from upstream packages: `git` (GPL-2.0), Python standard library
(PSF), `sglang` / `vLLM` (Apache-2.0).
