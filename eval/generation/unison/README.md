# Unison -- Evaluation Guide

## Overview

Unison evaluates Unified Multimodal Models (UMMs) by leveraging the synergy between understanding and generation capabilities across four dimensions:

- **IC** (Internal Consistency) -- generate an image from a prompt, then verify that the model's own understanding of that image (via yes/no probes against the reference) is internally consistent.
- **UGG** (Understanding-Guided Generation) -- predict a bounding box for an edit target from an instruction, then edit the image using the model's own predicted box.
- **GGU** (Generation-Guided Understanding) -- answer a spatial-reasoning question directly, generate an illustrative image from a description, then re-answer the question grounded in both the original and generated image.
- **ME** (Mutual Enhancement) -- two self-refining dialogues per sample: iterative editing + self-critique (u2g), and iterative mismatch detection + editing (g2u).

Scoring is performed by **Unison-Judge**, a Qwen3-VL-8B-Instruct-based evaluator that achieves 88.7% alignment with human judgments. It can run locally on GPUs or via any OpenAI-compatible API.

Reference: https://github.com/FudanCVL/Unison

Paper: https://arxiv.org/abs/2606.26984

## Prerequisites

### 1. Dataset

Download Unison-Bench from HuggingFace:

```bash
huggingface-cli download FudanCVL/Unison \
    --repo-type dataset --local-dir ${UMM_DATASETS}/unison/Unison-Bench/data
```

Dataset structure:
```
${UMM_DATASETS}/unison/Unison-Bench/data/
├── Internal_Consistency/     # prompts.txt, questions.json, images/
├── Und_Guided_Gen/           # UGG.csv + referenced images
├── Gen_Guided_Und/           # 2D_Spatial/, 3D_Spatial/, Complex_Relation/
└── Mutual_Enhancement/       # ME.csv + images/
```

### 2. Generation Model

Download the generation backbone (e.g., BAGEL) using TorchUMM's usual model download flow (see the main [data preparation guide](../../../docs/data-preparation.md)).

### 3. Unison-Judge (scoring model)

Download the Unison-Judge checkpoint:

```bash
huggingface-cli download FudanCVL/Unison-Judge \
    --local-dir ${UMM_MODEL_CACHE}/evaluator/Unison-Judge
```

This is the default local judge path used by `evaluate_unison.py`. No local weights are needed if you use the `api` judge backend instead (set `unison.scoring.judge_backend: api` in the config and provide `unison.scoring.api_key`, or set `OPENAI_API_KEY`).

## Evaluation Pipeline

Unison uses a two-stage evaluation:

### Stage 1: Generation

The generation model produces outputs for each task (edited/generated images + text answers):

```bash
PYTHONPATH=src python -m umm.cli.main eval \
    --config configs/eval/unison/unison_bagel_generate.yaml
```

Output structure:
```
output/unison/bagel/
├── IC/{case_00000, ...}/  + IC_bagel_results.csv
├── UGG/{case_00000, ...}/ + UGG_bagel_results.csv
├── GGU/{case_00000, ...}/ + GGU_bagel_results.csv
├── ME/{case_00000, ...}/  + ME_bagel_results.csv
└── overall_summary.json
```

Each `*_results.csv` uses the schema expected by the (ported) task evaluators under `tasks/`: `dialogue_index, image_path, input_data, prompt_used, model_response, status, task_id, model_name, task_name, operation_type, operation_index, round_number, question_id, question_text`.

To run only a subset of tasks, set `unison.tasks` in the config to a list (e.g. `[IC, UGG]`) or a comma-separated string instead of `all`.

### Stage 2: Scoring

Unison-Judge scores the generated outputs against ground truth:

```bash
PYTHONPATH=src python -m umm.cli.main eval \
    --config configs/eval/unison/unison_bagel_score.yaml
```

Scoring output: a single JSON file (`unison.score_output_path`) with `understanding_score` / `generation_score` / `unified_score` per task plus an overall score, e.g.:

```json
{
  "model_name": "bagel",
  "summary": {
    "overall_score": 53.2,
    "tasks": {
      "IC":  {"understanding_score": 96.0, "generation_score": 82.5, "unified_score": 80.3},
      "UGG": {"understanding_score": 57.6, "generation_score": 78.1, "unified_score": 67.9},
      "GGU": {"understanding_score": 28.2, "generation_score": 41.6, "unified_score": 32.0},
      "ME":  {"understanding_score": 7.2,  "generation_score": 57.7, "unified_score": 32.5}
    }
  },
  "stats": {...},
  "details": {...}
}
```

Since the local judge (Qwen3-VL-8B-Instruct-based) may need a different set of dependencies than a given generation backbone's environment, `unison.scoring.python_executable` can point the scoring step at a separate Python interpreter.

## Config Reference

Configs are in `configs/eval/unison/`. Key fields under the `unison` block:

```yaml
unison:
  data_root: ${UMM_DATASETS}/unison/Unison-Bench/data
  tasks: all                       # all | IC,UGG,GGU,ME | [IC, UGG, GGU, ME]
  out_dir: output/unison/bagel
  max_samples: 0                   # 0 = all
  resume: true                     # skip cases with a _done.ok marker
  max_iterations: 5                # ME dialogue rounds cap (1-5)
  mode: generate                   # generate | score | full
  score_output_path: output/unison/eval/bagel.json
  scoring:
    judge_backend: local           # local | api
    local_model_path: ${UMM_MODEL_CACHE}/evaluator/Unison-Judge
    gpu_ids: "0-7"
    judge_io_log: judge_io_local.csv
    max_workers: 8
    max_items_per_task: 0          # 0 = all
    pixel_space_bbox: false        # true for models emitting absolute pixel bboxes (e.g. OmniGen2, UniWorld-V1)
    imgedit_bbox_mode: full        # full | noscope
    # API backend alternative:
    # judge_backend: api
    # api_key: ${OPENAI_API_KEY}
    # api_model: gpt-4o
    # thinking_tasks: UGG,ME
```

## Attribution

`common/`, `tasks/`, `evaluate_unison.py`, `aggregate_results.py`, and the prompt templates / ME dialogue-control logic embedded in `run_generation.py` are ported (largely verbatim) from the official [Unison](https://github.com/FudanCVL/Unison) repository so that TorchUMM's outputs remain compatible with Unison-Judge's calibration. `run_generation.py` and `run_scoring.py` are new, adapting Unison's original inference pipeline to TorchUMM's `InferencePipeline` / YAML config conventions.
