# Reproducing the main SWE-Git-Bench numbers

## 1. Get the dataset

Download `SWE-Git-Bench-conflicts-v1.0.tar.gz`,
`SWE-Git-Bench-predictions-v1.0.tar.gz`, and
`SWE-Git-Bench-analysis-v1.0.tar.gz` from the Harvard Dataverse record:

https://doi.org/10.7910/DVN/FSFEZX

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

## 3. Generate new predictions

Use the prompt builders in `prompts/templates.py` with any model endpoint, then
write one JSON object per line with these fields:

```json
{"dataset":"lite","model_name":"example-model","conflict_id":"...","file_path":"...","raw_prediction":"..."}
```

The original sweep used temperature 0, a 30-minute per-call budget, and up to
two retries. Exact internal serving endpoints are not part of the release; the
prompt builders and scoring code reproduce the public protocol independently
of a specific serving stack.

## 4. Recompute statistical analyses

```bash
python analysis/bootstrap_ci.py
python analysis/irr_kappa.py \
    --judge1-file data/analysis/failure_taxonomy_deepseek_v32.jsonl \
    --judge2-file data/analysis/failure_taxonomy_gemma3_27b.jsonl
```

## Hardware notes

- Models $\le$14B run on a single A800 80GB.
- DeepSeek-V3, R1, Qwen3-235B-A22B, Qwen3-Coder-480B-A35B, GLM-5,
  Kimi, MiniMax run TP=4 or TP=8 across A800 nodes.
- Closed APIs (Claude-Sonnet-4.6, Gemini-3-Pro, GPT-5.4) consume
  approximately 20–40M input tokens and 4–8M output tokens in aggregate.

See paper Appendix `app:compute` for the full sweep cost ranges.
