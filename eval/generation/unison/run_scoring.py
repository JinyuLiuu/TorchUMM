#!/usr/bin/env python3
"""Unison scoring-only script.

Usage:
    python eval/generation/unison/run_scoring.py --config configs/eval/unison/unison_bagel_score.yaml

Calls ``evaluate_unison.py`` (ported from the official Unison benchmark) with
arguments derived from the ``unison`` block of a UMM YAML config. Can be run
in a separate Python environment that has the judge dependencies installed
(transformers, qwen_vl_utils, etc. for the local judge; openai for the API
judge).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from umm.core.config import load_config


def _resolve_path(path_str: str, repo_root: Path) -> Path:
    path_str = os.path.expandvars(str(path_str))
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path


def _normalize_backbone_name(name: str) -> str:
    normalized = name.strip().lower().replace("-", "_")
    aliases = {
        "showo2": "show_o2",
        "showo": "show_o2",
        "janus": "janus_pro",
        "januspro": "janus_pro",
        "omnigen": "omnigen2",
        "blip3": "blip3o",
        "blip3_o": "blip3o",
        "token_flow": "tokenflow",
    }
    return aliases.get(normalized, normalized)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Unison scoring by invoking evaluate_unison.py."
    )
    parser.add_argument(
        "--config", required=True,
        help="UMM YAML config containing `inference` and `unison` blocks.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    raw_cfg = load_config(args.config)

    inference_cfg = raw_cfg.get("inference", {})
    if not isinstance(inference_cfg, dict):
        inference_cfg = {}

    unison_cfg = raw_cfg.get("unison", {})
    if not isinstance(unison_cfg, dict):
        unison_cfg = {}

    backbone_raw = inference_cfg.get("backbone", "")
    backbone = _normalize_backbone_name(str(backbone_raw))

    out_dir = _resolve_path(
        str(unison_cfg.get("out_dir", f"output/unison/{backbone}")),
        repo_root,
    )
    # run_generation.py writes results directly under `out_dir` (which acts
    # as the model's result directory, i.e. `--result-dir` for evaluate_unison.py).
    result_dir = out_dir

    data_root_value = unison_cfg.get("data_root")
    if not data_root_value:
        print("[scoring] ERROR: `unison.data_root` is required.")
        return 1
    data_root = str(_resolve_path(str(data_root_value), repo_root))

    scoring_cfg = unison_cfg.get("scoring", {})
    if not isinstance(scoring_cfg, dict):
        scoring_cfg = {}

    tasks_value = unison_cfg.get("tasks", "all")
    if tasks_value in ("all", None):
        tasks_str = "IC,UGG,GGU,ME"
    elif isinstance(tasks_value, str):
        tasks_str = ",".join(t.strip().upper() for t in tasks_value.split(",") if t.strip())
    elif isinstance(tasks_value, list):
        tasks_str = ",".join(str(t).strip().upper() for t in tasks_value if str(t).strip())
    else:
        tasks_str = "IC,UGG,GGU,ME"

    output_path = _resolve_path(
        str(unison_cfg.get("score_output_path", out_dir.parent / "eval" / f"{out_dir.name}.json")),
        repo_root,
    )

    eval_script = Path(__file__).resolve().parent / "evaluate_unison.py"
    if not eval_script.exists():
        print(f"[scoring] ERROR: evaluate_unison.py not found at {eval_script}")
        return 1

    cmd = [
        sys.executable,
        str(eval_script),
        "--result-dir", str(result_dir),
        "--data-dir", data_root,
        "--inference-base-dir", str(result_dir),
        "--output", str(output_path),
        "--tasks", tasks_str,
    ]

    judge_backend = str(scoring_cfg.get("judge_backend", "local")).strip().lower()
    cmd.extend(["--judge-backend", judge_backend])

    if judge_backend == "local":
        local_model_path = scoring_cfg.get("local_model_path") or scoring_cfg.get("judge_model_path")
        if local_model_path:
            cmd.extend(["--local-model-path", os.path.expandvars(str(local_model_path))])
        gpu_ids = scoring_cfg.get("gpu_ids")
        if gpu_ids is not None:
            cmd.extend(["--gpu-ids", str(gpu_ids)])
        judge_io_log = scoring_cfg.get("judge_io_log")
        if judge_io_log is not None:
            cmd.extend(["--judge-io-log", str(judge_io_log)])
    else:
        api_key = scoring_cfg.get("api_key")
        if api_key:
            cmd.extend(["--api-key", str(api_key)])
        api_model = scoring_cfg.get("api_model") or scoring_cfg.get("model")
        if api_model:
            cmd.extend(["--model", str(api_model)])
        thinking_tasks = scoring_cfg.get("thinking_tasks")
        if thinking_tasks is not None:
            if isinstance(thinking_tasks, list):
                thinking_tasks = ",".join(str(t) for t in thinking_tasks)
            cmd.extend(["--thinking-tasks", str(thinking_tasks)])

    max_workers = scoring_cfg.get("max_workers")
    if max_workers is not None:
        cmd.extend(["--max-workers", str(int(max_workers))])

    max_items = scoring_cfg.get("max_items_per_task") or unison_cfg.get("max_samples")
    if max_items:
        cmd.extend(["--max-items", str(int(max_items))])

    pixel_space_bbox = scoring_cfg.get("pixel_space_bbox")
    if pixel_space_bbox:
        cmd.append("--pixel-space-bbox")

    imgedit_bbox_mode = scoring_cfg.get("imgedit_bbox_mode")
    if imgedit_bbox_mode:
        cmd.extend(["--imgedit-bbox-mode", str(imgedit_bbox_mode)])

    env = os.environ.copy()
    print(f"[scoring] running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=eval_script.parent, env=env)
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
