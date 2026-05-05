# Reproducing the main SWE-Git-Bench numbers

## 1. Get the dataset

Download `SWE-Git-Bench-conflicts-v1.0.tar.gz`,
`SWE-Git-Bench-predictions-v1.0.tar.gz`, and
`SWE-Git-Bench-analysis-v1.0.tar.gz` from the Harvard Dataverse anonymized
preview link:

https://dataverse.harvard.edu/previewurl.xhtml?token=cbf161fa-dc18-45d4-9678-5085aafe5092

Unpack into `./data/`.

## 2. Score submitted predictions

The Dataverse prediction archive contains the JSONL outputs used in the paper.
To score a new prediction file with rows of the form
`{dataset, model_name, conflict_id, file_path, raw_prediction}`:

```bash
python score.py \
    --eval-dataset data/conflicts/eval_dataset_lite.jsonl \
                   data/conflicts/eval_dataset_Verified.jsonl \
                   data/conflicts/eval_dataset_Multilingual.jsonl \
                   data/conflicts/eval_dataset_Multimodal.jsonl \
    --predictions my_predictions.jsonl \
    --output-csv scored_predictions.csv
```

The script reports exact match (EM), edit similarity (ES), line similarity
(LS), and block accuracy where applicable.

## 3. Serve open-weight models

For each open-weight model (TP$\le$8 on A100/H100 nodes):

```bash
python -m sglang.launch_server \
    --model-path <model> \
    --tp <degree> \
    --enable-metrics \
    --trust-remote-code \
    --port <port>
```

Each model exposes an OpenAI-compatible endpoint that the harness consumes.
Closed systems use their own vendor endpoints; the harness reads
`OPENAI_API_KEY`-style env vars per provider.

## 4. Run a new sweep

```bash
./run_eval_task.sh \
    --datasets "lite Verified Multilingual Multimodal" \
    --formats  "whole_file conflict_block" \
    --temperature 0 \
    --budget-sec 1800 \
    --max-retries 2
```

The original sweep writes JSONL files with predictions and EM/ES/LS/BA scores.
Exact internal serving endpoints are intentionally omitted from the anonymous
mirror; the prompts and scoring code are sufficient to reproduce the protocol
with any OpenAI-compatible model endpoint.

## 5. Recompute statistical analyses

```bash
python analysis/bootstrap_ci.py
python analysis/irr_kappa.py \
    --judge1-file data/analysis/failure_taxonomy_deepseek_v32.jsonl \
    --judge2-file data/analysis/failure_taxonomy_gemma3_27b.jsonl
```

## Hardware notes

- Models $\le$14B run on a single A100 80GB.
- DeepSeek-V3, R1, Qwen3-235B-A22B, Qwen3-Coder-480B-A35B, GLM-5,
  Kimi, MiniMax run TP=4 or TP=8 on one or two H100 nodes.
- Closed APIs (Claude-Sonnet-4.6, Gemini-3-Pro, GPT-5.4) consume
  approximately 20–40M input tokens and 4–8M output tokens in aggregate.

See paper Appendix `app:compute` for the full sweep cost ranges.
