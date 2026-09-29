# SWE-Git-Bench

Public code release for the **SWE-Git-Bench** benchmark scoring and analysis harness.

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
itself (canonical conflict JSONL files, ground-truth resolutions, prediction
matrix, analysis archive, Croissant core + RAI metadata, and the per-instance
manifest) is served from the companion Harvard Dataverse record:
https://doi.org/10.7910/DVN/FSFEZX

## License

CC BY 4.0 for the harness and prompt templates. Per-component licenses
inherit from upstream packages: `git` (GPL-2.0), Python standard library
(PSF), `sglang` / `vLLM` (Apache-2.0).
