#!/usr/bin/env python3
"""
Unison evaluation pipeline — all four tasks (IC / UGG / GGU / ME). The judge
runs either locally (a trained Unison-Judge checkpoint, default) or via any
OpenAI-compatible API.

Ported from the official Unison benchmark (FudanCVL/Unison) so TorchUMM can
score CSVs produced by ``eval/generation/unison/run_generation.py``. The task
evaluators (``tasks/``) and shared judge/IO utilities (``common/``) are
unchanged from upstream; only this entry point's default paths / docstring
were adapted for TorchUMM's layout.

Upstream: https://github.com/FudanCVL/Unison

Usage
-----
# local judge (default) — point at the Unison-Judge weights
python evaluate_unison.py \
    --result-dir  output/unison/bagel \
    --data-dir    ${UMM_DATASETS}/unison/data \
    --inference-base-dir output/unison/bagel \
    --judge-backend local --local-model-path ${UMM_MODEL_CACHE}/evaluator/Unison-Judge --gpu-ids 0,1,2,3 \
    --output      output/unison/eval/bagel.json

# API judge
python evaluate_unison.py ... --judge-backend api --api-key sk-xxxx

--result-dir should contain subdirectories IC/, UGG/, GGU/, ME/ with
*_results.csv files produced by run_generation.py.
"""

import argparse
import json
import os
import sys

from common.judge import ClosedSourceJudge
from tasks.evaluate_ic import evaluate_ic
from tasks.evaluate_ugg import evaluate_ugg
from tasks.evaluate_ggu import evaluate_ggu
from tasks.evaluate_me import evaluate_me

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "evaluation_results")

# Models that output absolute pixel coordinates for UGG bbox prediction.
# All others are treated as relative (0-1 or 0-1000) coordinates.
PIXEL_SPACE_BBOX_MODELS = {"omnigen2", "uniworld_v1", "uniworld-v1"}


def _result_csv_path(model_name: str, task_id: str) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return os.path.join(RESULTS_DIR, f"{model_name}_{task_id}.csv")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_csv(task_dir: str, task_id: str) -> str:
    """Return path to the merged *_results.csv inside task_dir, or '' if not found."""
    if not os.path.isdir(task_dir):
        return ""
    candidates = [f for f in os.listdir(task_dir)
                  if f.endswith("_results.csv") and task_id in f]
    merged = [f for f in candidates if "_gpu" not in f]
    for fname in sorted(merged) or sorted(candidates):
        return os.path.join(task_dir, fname)
    for fname in sorted(os.listdir(task_dir)):
        if fname.endswith(".csv"):
            return os.path.join(task_dir, fname)
    return ""


def resolve_local_io_log(path: str) -> str:
    """Resolve the local-judge I/O log path. Empty -> disabled ("")."""
    if not path:
        return ""
    abs_path = os.path.abspath(path)
    if os.path.basename(abs_path) == "judge_io.csv":
        raise SystemExit(
            "[ERROR] --judge-io-log must not be named judge_io.csv "
            "(reserved for the SFT capture set). Use e.g. judge_io_local.csv."
        )
    return abs_path


def _overall_score(task_results: dict):
    scores = []
    for task_id, result in task_results.items():
        uni = result["scores"].get("unified_score")
        if uni is not None:
            scores.append(uni)
    if not scores:
        return None
    return round(sum(scores) / len(scores), 6)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unison evaluation pipeline")
    p.add_argument(
        "--result-dir", required=True,
        help="Path to model result directory (contains IC/, UGG/, GGU/, ME/ subdirs)",
    )
    p.add_argument(
        "--data-dir", default=None,
        help="Path to Unison benchmark data directory (contains "
             "Internal_Consistency/, Und_Guided_Gen/, Gen_Guided_Und/, Mutual_Enhancement/).",
    )
    p.add_argument(
        "--inference-base-dir", default=None,
        help="Base directory for resolving relative image paths in CSVs. "
             "Defaults to --result-dir (TorchUMM's run_generation.py writes absolute paths).",
    )
    p.add_argument(
        "--api-key", default=None,
        help="API key for the closed-source judge. Falls back to OPENAI_API_KEY env var.",
    )
    p.add_argument(
        "--model", default="gpt-4o",
        help="Closed-source judge model name (default: gpt-4o)",
    )
    p.add_argument(
        "--output", default=None,
        help="Output JSON path. Defaults to eval_<model_name>.json in current dir.",
    )
    p.add_argument(
        "--tasks", default="IC,UGG,GGU,ME",
        help="Comma-separated list of tasks to evaluate (default: IC,UGG,GGU,ME)",
    )
    p.add_argument(
        "--max-workers", type=int, default=8,
        help="Max concurrent judge API threads (default: 8)",
    )
    p.add_argument(
        "--max-items", type=int, default=None,
        help="Limit number of items per task (for quick testing)",
    )
    p.add_argument(
        "--pixel-space-bbox", action="store_true", default=None,
        help="Force UGG bbox to be treated as absolute pixel coordinates. "
             "Auto-detected from model name if not set.",
    )
    p.add_argument(
        "--judge-backend", choices=["local", "api"], default="local",
        help="Judge backend: 'local' (Unison-Judge via GPUs, default) or "
             "'api' (OpenAI-compatible closed-source model).",
    )
    p.add_argument(
        "--thinking-tasks", default="UGG,ME",
        help="Comma-separated tasks for which the API judge runs in thinking mode "
             "(default: UGG,ME). Only applies to --judge-backend api; empty disables.",
    )
    p.add_argument(
        "--imgedit-bbox-mode", choices=["full", "noscope"], default="full",
        help="ImgEdit (UGG / ME-generation) bbox handling. 'full' (default) embeds the "
             "GT region-scoping instruction; 'noscope' drops the bbox AND removes the "
             "'Evaluation scope:' line so only ME/misalignment uses bbox. Applies to "
             "both judge backends.",
    )
    p.add_argument(
        "--local-model-path",
        default=None,
        help="Path to the trained local judge model / Unison-Judge weights "
             "(used when --judge-backend local). Defaults to "
             "${UMM_MODEL_CACHE}/evaluator/Unison-Judge if set, else ./unison-judge.",
    )
    p.add_argument(
        "--gpu-ids", default="0-7",
        help="GPUs for the local judge, e.g. '0-7' or '0,1,2' (default: 0-7).",
    )
    p.add_argument(
        "--judge-io-log", default="judge_io_local.csv",
        help="Local-judge per-call I/O log path (local backend only). "
             "Empty string disables. Must NOT be judge_io.csv.",
    )
    return p.parse_args()


def _default_local_model_path() -> str:
    model_cache = os.environ.get("UMM_MODEL_CACHE")
    if model_cache:
        return os.path.join(model_cache, "evaluator", "Unison-Judge")
    return os.path.join(os.path.dirname(__file__), "unison-judge")


def main():
    args = parse_args()

    result_dir = os.path.abspath(args.result_dir)
    model_name = os.path.basename(result_dir)

    if args.data_dir:
        data_dir = os.path.abspath(args.data_dir)
    else:
        data_dir = ""
        print("[WARN] --data-dir not provided; GT data won't be loaded for GGU/UGG/ME/IC.")

    if args.inference_base_dir:
        inference_base_dir = os.path.abspath(args.inference_base_dir)
    else:
        # run_generation.py always writes absolute paths, so this only matters
        # as a fallback for any residual relative paths.
        inference_base_dir = result_dir

    if args.pixel_space_bbox is not None:
        pixel_space_bbox = args.pixel_space_bbox
    else:
        pixel_space_bbox = model_name.strip().lower() in PIXEL_SPACE_BBOX_MODELS

    output_path = args.output or f"eval_{model_name}.json"

    print(f"Model:              {model_name}")
    print(f"Result dir:         {result_dir}")
    print(f"Data dir:           {data_dir or '(none)'}")
    print(f"Inference base dir: {inference_base_dir}")
    print(f"Output:             {output_path}")
    print(f"UGG bbox mode:      {'pixel (absolute)' if pixel_space_bbox else 'relative (0-1 / 0-1000)'}")
    print()

    # Build judge
    if args.judge_backend == "local":
        io_log = resolve_local_io_log(args.judge_io_log)
        if io_log:
            os.environ["JUDGE_IO_LOG"] = io_log
            print(f"Local judge I/O log: {io_log}")
        else:
            os.environ.pop("JUDGE_IO_LOG", None)
        from common.local_judge import LocalQwenVLJudge
        local_model_path = args.local_model_path or _default_local_model_path()
        print(f"Judge backend:      local ({local_model_path}) gpus={args.gpu_ids}")
        judge = LocalQwenVLJudge(local_model_path, args.gpu_ids, max_workers=args.max_workers)
    else:
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            print("[ERROR] No API key. Pass --api-key or set OPENAI_API_KEY.",
                  file=sys.stderr)
            sys.exit(1)
        print(f"Judge backend:      api ({args.model})")
        judge = ClosedSourceJudge(api_key=api_key, model=args.model, max_workers=args.max_workers)

    judge.imgedit_bbox_mode = args.imgedit_bbox_mode
    print(f"ImgEdit bbox mode:  {args.imgedit_bbox_mode}"
          + ("  (only ME/misalignment uses bbox)" if args.imgedit_bbox_mode == "noscope" else ""))

    thinking_tasks = {t.strip().upper() for t in args.thinking_tasks.split(",") if t.strip()}
    if args.judge_backend == "api" and thinking_tasks:
        print(f"API judge thinking mode: {sorted(thinking_tasks)}")

    tasks_to_run = [t.strip().upper() for t in args.tasks.split(",")]
    task_data_dirs = {
        "IC":  os.path.join(data_dir, "Internal_Consistency"),
        "UGG": os.path.join(data_dir, "Und_Guided_Gen"),
        "GGU": os.path.join(data_dir, "Gen_Guided_Und"),
        "ME":  os.path.join(data_dir, "Mutual_Enhancement"),
    }
    task_evaluators = {
        "IC":  evaluate_ic,
        "UGG": evaluate_ugg,
        "GGU": evaluate_ggu,
        "ME":  evaluate_me,
    }

    task_results = {}

    for task_id in tasks_to_run:
        if task_id not in task_evaluators:
            print(f"[WARN] Unknown task: {task_id}, skipping.")
            continue

        task_result_dir = os.path.join(result_dir, task_id)
        csv_path = _find_csv(task_result_dir, task_id)
        task_data_dir = task_data_dirs.get(task_id, "")

        print(f"{'='*60}")
        print(f"Task: {task_id}")
        print(f"  CSV:      {csv_path or '(not found)'}")
        print(f"  Data dir: {task_data_dir}")

        evaluator = task_evaluators[task_id]
        out_csv = _result_csv_path(model_name, task_id)
        extra = {"pixel_space_bbox": pixel_space_bbox} if task_id == "UGG" else {}

        if args.judge_backend == "api" and hasattr(judge, "enable_thinking"):
            judge.enable_thinking = task_id in thinking_tasks
            print(f"  Judge thinking:  {'ON' if judge.enable_thinking else 'off'}")
        result = evaluator(
            csv_path=csv_path,
            data_dir=task_data_dir,
            judge=judge,
            inference_base_dir=inference_base_dir,
            max_workers=args.max_workers,
            output_csv=out_csv,
            **extra,
        )
        print(f"  Saved: {out_csv}")
        task_results[task_id] = result

        scores = result["scores"]
        stats = result["stats"]
        print(
            f"  understanding={scores['understanding_score']}, "
            f"generation={scores['generation_score']}, "
            f"unified={scores['unified_score']} "
            f"[{stats['num_scored']}/{stats['num_total']} scored]"
        )

    print(f"{'='*60}")
    overall = _overall_score(task_results)
    print(f"Overall score (mean of task unified scores): {overall}")

    output = {
        "model_name": model_name,
        "summary": {
            "overall_score": overall,
            "tasks": {
                task_id: result["scores"]
                for task_id, result in task_results.items()
            },
        },
        "stats": {
            task_id: result["stats"]
            for task_id, result in task_results.items()
        },
        "details": {
            task_id: result["details"]
            for task_id, result in task_results.items()
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {os.path.abspath(output_path)}")

    if hasattr(judge, "close"):
        judge.close()


if __name__ == "__main__":
    main()
