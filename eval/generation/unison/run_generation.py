#!/usr/bin/env python3
"""Unison generation-only evaluation script.

Usage:
    python eval/generation/unison/run_generation.py --config configs/eval/unison/unison_bagel.yaml

This script handles ONLY the generation phase of Unison evaluation. It loads
a UMM YAML config, creates an ``InferencePipeline``, dispatches to the four
Unison tasks (IC, UGG, GGU, ME), and writes per-task CSVs in the exact schema
expected by the ported evaluators under ``tasks/`` (see ``run_scoring.py``
and ``evaluate_unison.py``).

Task semantics follow the official Unison benchmark
(https://github.com/FudanCVL/Unison, https://arxiv.org/abs/2606.26984):

- IC  (Internal Consistency):          generate an image from a prompt, then
                                        answer per-question yes/no probes
                                        about the *reference* image; the
                                        scorer separately asks a judge the
                                        same questions about the *generated*
                                        image.
- UGG (Understanding-Guided Generation): predict a bbox for the edit target
                                        from the instruction, then edit the
                                        image using the model's own predicted
                                        bbox.
- GGU (Generation-Guided Understanding): answer a spatial-reasoning question
                                        directly, generate an illustrative
                                        image from its description, then
                                        re-answer the question grounded in
                                        both the original and generated image.
- ME  (Mutual Enhancement):            two self-refining dialogues per
                                        sample -- u2g (iterative editing +
                                        self-critique) and g2u (iterative
                                        mismatch detection + editing).
"""
from __future__ import annotations

# fmt: off
import argparse
import csv
import json
import re
import shutil
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from umm.core.config import load_config
from umm.inference import InferencePipeline

# fmt: on


# =============================================================================
# Context API -- mirrors eval/generation/uni_mmmu/run_generation.py
# =============================================================================

@dataclass
class CtxImagePath:
    path: str
    mime: str = "image/png"


@dataclass
class ContextItem:
    kind: Literal["text", "image"]
    payload: Union[str, CtxImagePath]


def add_text(ctx: List[ContextItem], text: str) -> None:
    ctx.append(ContextItem("text", text))


def add_image_path(ctx: List[ContextItem], path: Union[str, Path], mime: str = "image/png") -> None:
    ctx.append(ContextItem("image", CtxImagePath(str(path), mime)))


def _extract_text(output: Any) -> str:
    if isinstance(output, str):
        return output
    if isinstance(output, dict):
        for key in ("text", "answer", "response", "output", "generated_text"):
            value = output.get(key)
            if isinstance(value, str):
                return value
        results = output.get("results")
        if isinstance(results, dict):
            for key in ("text", "answer", "response", "output"):
                value = results.get(key)
                if isinstance(value, str):
                    return value
        if isinstance(results, list):
            for item in results:
                text = _extract_text(item)
                if text:
                    return text
        for list_key in ("understandings",):
            container = output.get(list_key)
            if isinstance(container, list):
                for item in container:
                    text = _extract_text(item)
                    if text:
                        return text
    if isinstance(output, list):
        for item in output:
            text = _extract_text(item)
            if text:
                return text
    return ""


def _extract_saved_path(result: Any, out_path: Path) -> str:
    if isinstance(result, dict):
        saved = result.get("saved_paths") or result.get("output_path")
        if isinstance(saved, list) and saved:
            return str(saved[0])
        if isinstance(saved, str) and saved:
            return saved
        img = result.get("image")
        if isinstance(img, Image.Image):
            out_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(str(out_path), format="PNG")
            return str(out_path)
        imgs = result.get("images")
        if isinstance(imgs, list) and imgs and isinstance(imgs[0], Image.Image):
            out_path.parent.mkdir(parents=True, exist_ok=True)
            imgs[0].save(str(out_path), format="PNG")
            return str(out_path)
    if isinstance(result, Image.Image):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(str(out_path), format="PNG")
        return str(out_path)
    if out_path.is_file() and out_path.stat().st_size > 0:
        return str(out_path)
    return ""


def _ctx_to_text_and_images(ctx: List[ContextItem], prompt_suffix: str = "") -> Tuple[str, List[str]]:
    text_parts: List[str] = []
    images: List[str] = []
    for item in ctx:
        if item.kind == "text":
            text_parts.append(str(item.payload))
        elif item.kind == "image":
            images.append(item.payload.path)  # type: ignore[union-attr]
    if prompt_suffix:
        text_parts.append(prompt_suffix)
    return "\n".join(text_parts), images


def generate_text_from_context(
    pipeline: InferencePipeline,
    backbone: str,
    ctx: List[ContextItem],
    prompt_suffix: str = "",
    params: Optional[Dict[str, Any]] = None,
) -> str:
    prompt, images = _ctx_to_text_and_images(ctx, prompt_suffix)
    payload: Dict[str, Any] = {
        "backbone": backbone,
        "task": "understanding",
        "prompt": prompt or "Describe what you see.",
        "images": images,
        "params": params or {},
    }
    try:
        result = pipeline.run(payload)
    except (ValueError, NotImplementedError) as exc:
        if "image" in str(exc).lower() or isinstance(exc, NotImplementedError):
            return ""
        raise
    return _extract_text(result)


def generate_image_from_context(
    pipeline: InferencePipeline,
    backbone: str,
    ctx: List[ContextItem],
    out_path: Union[str, Path],
    prompt_suffix: str = "",
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Any]:
    """Generate an image conditioned on the accumulated context.

    Tries ``task="editing"`` when context contains images (image-conditioned
    generation), falling back to ``task="generation"`` (text-only) if the
    backbone does not implement editing. Returns ``(saved_image_path, raw_result)``.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    workspace = out_path.parent / "_gen_workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    prompt, images = _ctx_to_text_and_images(ctx, prompt_suffix)
    result: Any = None
    workspace_file = str(workspace / out_path.name)

    if images:
        try:
            payload: Dict[str, Any] = {
                "backbone": backbone,
                "task": "editing",
                "prompt": prompt or "Generate an image.",
                "images": images,
                "output_path": workspace_file,
                "params": params or {},
            }
            result = pipeline.run(payload)
        except (NotImplementedError, ValueError, RuntimeError, TypeError):
            result = None

    if result is None:
        payload = {
            "backbone": backbone,
            "task": "generation",
            "prompt": prompt or "Generate an image.",
            "output_path": workspace_file,
            "params": params or {},
        }
        result = pipeline.run(payload)

    subprocess_failed = isinstance(result, dict) and (
        result.get("returncode") not in (None, 0)
        or (result.get("error") and not result.get("images"))
    )
    if subprocess_failed:
        stderr_tail = (result.get("stderr") or "")[-800:]
        rc = result.get("returncode", "?")
        err_msg = result.get("error", "")
        print(f"[gen] backbone failed (rc={rc}): {err_msg}\n{stderr_tail}")

    saved = _extract_saved_path(result, workspace / out_path.name)
    if not saved:
        img_exts = {".png", ".jpg", ".jpeg", ".webp"}
        candidates = sorted(
            [f for f in workspace.rglob("*") if f.is_file() and f.suffix.lower() in img_exts],
            key=lambda p: p.stat().st_mtime,
        )
        if candidates:
            saved = str(candidates[0])

    if saved and Path(saved).is_file():
        shutil.copy2(saved, str(out_path))
        saved = str(out_path)

    if not saved and subprocess_failed:
        rc = result.get("returncode", "?")
        stderr_tail = (result.get("stderr") or "").strip().splitlines()
        hint = stderr_tail[-1] if stderr_tail else "unknown error"
        raise RuntimeError(f"Image generation subprocess failed (rc={rc}) and produced no image: {hint}")

    return saved, result


# =============================================================================
# ME dialogue-control helpers, ported from Unison's Inference_Pipeline/infer.py
# (https://github.com/FudanCVL/Unison) -- kept verbatim so generated dialogues
# match the format the (also ported) tasks/evaluate_me.py parser expects.
# =============================================================================

def extract_combined_instructions(output: str) -> str:
    """Extract multiple instructions from a g2u understanding-round output."""
    pattern = r'\d+\.\[[^\]]+\]:\[([^\]]+)\]'
    matches = re.findall(pattern, output)
    if matches:
        return ', '.join(matches)
    if output.lower().startswith('no'):
        rest = output[2:].strip()
        while rest and rest[0] in ".,:;":
            rest = rest[1:].strip()
        if rest:
            return rest
    return output


def parse_evaluation_output(output: str) -> Tuple[bool, Optional[str], bool]:
    """Parse a u2g evaluation-round output: (should_stop, next_instruction, format_valid)."""
    output = output.strip()
    output_lower = output.lower()

    yes_match = re.search(r'\byes\b', output_lower)
    no_match = re.search(r'\bno\b', output_lower)

    if yes_match and no_match:
        if yes_match.start() < no_match.start():
            return True, None, True
        extracted = extract_combined_instructions(output)
        if extracted != output and extracted:
            return False, extracted, True
        no_start = no_match.start()
        rest = output[no_start + 2:].strip()
        while rest and rest[0] in ".,:;":
            rest = rest[1:].strip()
        return False, rest if rest else None, True
    if yes_match:
        return True, None, True
    if no_match:
        extracted = extract_combined_instructions(output)
        if extracted != output and extracted:
            return False, extracted, True
        no_start = no_match.start()
        rest = output[no_start + 2:].strip()
        while rest and rest[0] in ".,:;":
            rest = rest[1:].strip()
        return False, rest if rest else None, True
    return False, None, False


# =============================================================================
# Config / path helpers
# =============================================================================

def _resolve_path(path_str: str, repo_root: Path) -> Path:
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


def _load_eval_cfg(config_path: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    raw_cfg = load_config(config_path)
    eval_cfg = raw_cfg.get("eval", {}) if isinstance(raw_cfg.get("eval"), dict) else {}
    unison_cfg = raw_cfg.get("unison", {}) if isinstance(raw_cfg.get("unison"), dict) else {}
    inference_cfg = raw_cfg.get("inference", {}) if isinstance(raw_cfg.get("inference"), dict) else {}
    if not eval_cfg and "benchmark" in raw_cfg:
        eval_cfg = {"benchmark": raw_cfg.get("benchmark")}
    return eval_cfg, unison_cfg, inference_cfg


def _sanitize_name(s: str, maxlen: int = 120) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-\.]+", "_", (s or "").strip())[:maxlen] or "item"


def _is_cuda_fatal(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in ("device-side assert", "cuda error", "cublas", "cudnn"))


# =============================================================================
# CSV writer -- schema matches tasks/evaluate_*.py (ported from Unison's
# Evaluation_Pipeline, which reads inference CSVs with these exact columns).
# =============================================================================

_CSV_FIELDS = [
    "dialogue_index", "image_path", "input_data", "prompt_used",
    "model_response", "status", "task_id", "model_name", "task_name",
    "operation_type", "operation_index", "round_number",
    "question_id", "question_text",
]


class ResultWriter:
    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._is_new = not csv_path.exists() or csv_path.stat().st_size == 0
        self._fh = csv_path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=_CSV_FIELDS, extrasaction="ignore", restval="")
        if self._is_new:
            self._writer.writeheader()
            self._fh.flush()

    def write(self, row: Dict[str, Any]) -> None:
        self._writer.writerow(row)
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def _base_row(
    dialogue_index: int,
    image_path: str,
    input_data: Dict[str, Any],
    prompt_used: str,
    model_response: str,
    status: str,
    task_id: str,
    model_name: str,
    operation_type: str,
    operation_index: int,
    round_number: Any = "",
    question_id: str = "",
    question_text: str = "",
) -> Dict[str, Any]:
    return {
        "dialogue_index": dialogue_index,
        "image_path": image_path or "",
        "input_data": json.dumps(input_data, ensure_ascii=False),
        "prompt_used": prompt_used,
        "model_response": model_response,
        "status": status,
        "task_id": task_id,
        "model_name": model_name,
        "task_name": task_id,
        "operation_type": operation_type,
        "operation_index": operation_index,
        "round_number": round_number,
        "question_id": question_id,
        "question_text": question_text,
    }


# =============================================================================
# Data loading -- ported from Unison's Inference_Pipeline/infer.py so index
# semantics match the (also ported) GT loaders in tasks/../common/io.py.
# =============================================================================

def _load_ic_task_data(data_dir: Path) -> List[Dict[str, Any]]:
    prompts_file = data_dir / "prompts.txt"
    questions_file = data_dir / "questions.json"
    images_dir = data_dir / "images"

    prompts = [ln.strip() for ln in prompts_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    questions_data = json.loads(questions_file.read_text(encoding="utf-8"))

    data: List[Dict[str, Any]] = []
    for idx, prompt in enumerate(prompts):
        prompt_key = f"prompt_{idx}"
        questions = questions_data.get("questions", {}).get(prompt_key)
        if not questions:
            continue
        image_num = idx + 1
        img_path = None
        for ext in ("png", "jpg", "jpeg"):
            candidate = images_dir / f"{image_num}.{ext}"
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            continue
        data.append({
            "index": idx,
            "image_path": str(img_path),
            "prompt": prompt,
            "questions": questions,
        })
    print(f"[IC] loaded {len(data)}/{len(prompts)} items")
    return data


def _load_ugg_task_data(data_dir: Path) -> List[Dict[str, Any]]:
    df = pd.read_csv(data_dir / "UGG.csv")
    data: List[Dict[str, Any]] = []
    skipped = 0
    for idx, row in df.iterrows():
        image_path = data_dir / str(row["image_path"])
        if not image_path.exists():
            skipped += 1
            continue
        data.append({
            "index": int(idx),
            "image_path": str(image_path),
            "object": str(row.get("object", "")),
            "instruction": str(row.get("instruction", "")),
            "operation": str(row.get("operation", "")),
            "bbox": str(row.get("bbox", "")),
            "mask": str(row.get("mask", "")),
        })
    print(f"[UGG] loaded {len(data)}/{len(df)} items ({skipped} skipped, image not found)")
    return data


def _load_ggu_task_data(data_dir: Path) -> List[Dict[str, Any]]:
    sub_tasks = [
        ("2D_Spatial", "2d_spatial.json"),
        ("3D_Spatial", "spatial.json"),
        ("Complex_Relation", "complex_relation.json"),
    ]
    data: List[Dict[str, Any]] = []
    skipped = 0
    total = 0
    for category_dir, filename in sub_tasks:
        json_file = data_dir / category_dir / filename
        if not json_file.exists():
            print(f"[GGU] {json_file} not found, skipping")
            continue
        items = json.loads(json_file.read_text(encoding="utf-8"))
        category = category_dir.lower()
        total += len(items)
        for item in items:
            image_path_raw = item.get("image_path", "")
            image_path = ""
            if image_path_raw:
                candidate = data_dir / category_dir / image_path_raw
                if not candidate.exists():
                    skipped += 1
                    continue
                image_path = str(candidate)
            data.append({
                "index": len(data),
                "image_path": image_path,
                "category": category,
                "question": item.get("question", ""),
                "options": item.get("options", {}),
                "answer": item.get("answer", ""),
                "description": item.get("description", ""),
                "image_generation_validate": item.get("image_generation_validate", {}),
            })
    print(f"[GGU] loaded {len(data)}/{total} items ({skipped} skipped, image not found)")
    return data


def _load_me_task_data(data_dir: Path) -> List[Dict[str, Any]]:
    df = pd.read_csv(data_dir / "ME.csv")
    data: List[Dict[str, Any]] = []
    skipped = 0
    for idx, row in df.iterrows():
        rel_path = str(row["image_path"])
        image_path = data_dir / rel_path
        if not image_path.exists():
            image_path = data_dir / "images" / rel_path
        if not image_path.exists():
            skipped += 1
            continue
        data.append({
            "index": int(idx),
            "image_path": str(image_path),
            "operation": str(row.get("operation", "")),
            "instruction": str(row.get("instruction", "")),
            "caption": str(row.get("caption", "")),
            "final_caption": str(row.get("final_caption", "")),
        })
    print(f"[ME] loaded {len(data)}/{len(df)} items ({skipped} skipped, image not found)")
    return data


# =============================================================================
# Prompt templates -- text kept verbatim from Unison's
# Inference_Pipeline/prompt_templates.py so the (ported, unmodified) judge
# prompts and scoring regexes still apply to our outputs.
# =============================================================================

_ME_U2G_EVAL_TEMPLATE = (
    "Based on the original image, edited image and instruction: {instruction}\n"
    "Carefully compare the original image and the edited image. You must check each "
    "operation mentioned in the instruction one by one to verify if it is perfectly and "
    "completely satisfied in the edited image.\n"
    "If ALL operations are perfectly satisfied (no missing, incomplete, or incorrect parts), "
    "output 'Yes'.\n"
    "If ANY operation is not perfectly satisfied, incomplete, or incorrectly executed, you "
    "MUST output 'No, [edit instruction]' describing what needs to be fixed based on the "
    "unsatisfied part(s)."
)

_ME_G2U_UNDERSTANDING_TEMPLATE = (
    "Based on the image and caption: {caption}\n"
    "Identify any mismatches between the image and caption. Output 'Yes' if there are no "
    "mismatches. Output 'No, 1.[mismatch1]:[edit instruction1], "
    "2.[mismatch2]:[edit instruction2],...' If there are mismatches."
)


def _me_g2u_editing_prompt(combined_instructions: str) -> str:
    return f"Based on the instruction: {combined_instructions}, please perform image editing task."


def _ugg_understanding_prompt(instruction: str) -> str:
    return (
        f"Based on the image editing instruction: {instruction}, "
        "please output the target object location. "
        "Output only the bbox in the format [x_min, y_min, x_max, y_max]"
    )


def _ugg_editing_prompt(instruction: str, bbox: str) -> str:
    return f"Based on the target object location: {bbox} and instruction: {instruction}, please perform image editing task."


def _ggu_understanding_prompt(question: str, options: Dict[str, str]) -> str:
    options_text = "\n".join(f"{k}: {v}" for k, v in options.items())
    return f"{question}\n\nOptions:\n{options_text}\nAnswer ONLY option"


def _ggu_unify_prompt(question: str, options: Dict[str, str]) -> str:
    options_text = "\n".join(f"{k}: {v}" for k, v in options.items())
    return (
        "The second image is a reference generated based on the spatial requirements of the "
        "question. Reason both the original scene image and this reference image, answer the "
        f"following question:\n\n{question}\nOptions:\n{options_text}\nAnswer ONLY option"
    )


# =============================================================================
# TASK: IC (Internal Consistency)
# =============================================================================

def run_task_ic(
    pipeline: InferencePipeline,
    backbone: str,
    data_root: Path,
    out_root: Path,
    max_samples: int,
    resume: bool,
    request_params: Dict[str, Any],
    model_name: str,
) -> Dict[str, Any]:
    task_id = "IC"
    out_dir = out_root / task_id
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{task_id}_{model_name}_results.csv"
    writer = ResultWriter(csv_path)

    items_dir = data_root / "Internal_Consistency"
    cases = _load_ic_task_data(items_dir)
    if max_samples > 0:
        cases = cases[:max_samples]

    n_success = n_error = n_skipped = 0
    for case in tqdm(cases, desc=f"[{task_id}]"):
        idx = case["index"]
        item_dir = out_dir / f"case_{idx:05d}"
        item_dir.mkdir(parents=True, exist_ok=True)
        done_marker = item_dir / "_done.ok"
        if resume and done_marker.exists():
            n_skipped += 1
            continue

        try:
            # Generation: produce an image from the prompt alone.
            ctx: List[ContextItem] = []
            add_text(ctx, case["prompt"])
            img_out = item_dir / "model_image.png"
            img_path, _ = generate_image_from_context(
                pipeline, backbone, ctx, out_path=img_out, params=request_params,
            )
            writer.write(_base_row(
                dialogue_index=idx, image_path=case["image_path"], input_data=case,
                prompt_used=case["prompt"], model_response=img_path or "",
                status="success" if img_path else "error: no image produced",
                task_id=task_id, model_name=model_name,
                operation_type="generation", operation_index=1,
            ))

            # Understanding: yes/no probes against the *reference* image.
            for qid, question in case["questions"].items():
                q_prompt = f"{question} Answer ONLY yes or no."
                u_ctx: List[ContextItem] = []
                add_image_path(u_ctx, case["image_path"])
                add_text(u_ctx, q_prompt)
                answer = generate_text_from_context(pipeline, backbone, u_ctx, params=request_params)
                writer.write(_base_row(
                    dialogue_index=idx, image_path=case["image_path"], input_data=case,
                    prompt_used=q_prompt, model_response=answer or "",
                    status="success", task_id=task_id, model_name=model_name,
                    operation_type="understanding", operation_index=0,
                    question_id=str(qid), question_text=question,
                ))

            done_marker.write_text("ok", encoding="utf-8")
            n_success += 1
        except Exception as exc:
            n_error += 1
            print(f"[{task_id}] case {idx} failed: {exc}")
            traceback.print_exc(limit=2)
            if _is_cuda_fatal(exc):
                break

    writer.close()
    summary = {"task": task_id, "total": len(cases), "success": n_success, "error": n_error, "skipped": n_skipped}
    print(f"[{task_id}] {summary}")
    return summary


# =============================================================================
# TASK: UGG (Understanding-Guided Generation)
# =============================================================================

def run_task_ugg(
    pipeline: InferencePipeline,
    backbone: str,
    data_root: Path,
    out_root: Path,
    max_samples: int,
    resume: bool,
    request_params: Dict[str, Any],
    model_name: str,
) -> Dict[str, Any]:
    task_id = "UGG"
    out_dir = out_root / task_id
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{task_id}_{model_name}_results.csv"
    writer = ResultWriter(csv_path)

    cases = _load_ugg_task_data(data_root / "Und_Guided_Gen")
    if max_samples > 0:
        cases = cases[:max_samples]

    n_success = n_error = n_skipped = 0
    for case in tqdm(cases, desc=f"[{task_id}]"):
        idx = case["index"]
        item_dir = out_dir / f"case_{idx:05d}"
        item_dir.mkdir(parents=True, exist_ok=True)
        done_marker = item_dir / "_done.ok"
        if resume and done_marker.exists():
            n_skipped += 1
            continue

        try:
            # Understanding: predict the bbox of the edit target.
            u_prompt = _ugg_understanding_prompt(case["instruction"])
            u_ctx: List[ContextItem] = []
            add_image_path(u_ctx, case["image_path"])
            add_text(u_ctx, u_prompt)
            pred_bbox = generate_text_from_context(pipeline, backbone, u_ctx, params=request_params)
            writer.write(_base_row(
                dialogue_index=idx, image_path=case["image_path"], input_data=case,
                prompt_used=u_prompt, model_response=pred_bbox or "",
                status="success", task_id=task_id, model_name=model_name,
                operation_type="understanding", operation_index=0,
            ))

            # Editing: edit using the model's OWN predicted bbox (not GT).
            e_prompt = _ugg_editing_prompt(case["instruction"], pred_bbox or "")
            e_ctx: List[ContextItem] = []
            add_image_path(e_ctx, case["image_path"])
            add_text(e_ctx, e_prompt)
            img_out = item_dir / "model_image.png"
            img_path, _ = generate_image_from_context(
                pipeline, backbone, e_ctx, out_path=img_out, params=request_params,
            )
            writer.write(_base_row(
                dialogue_index=idx, image_path=case["image_path"], input_data=case,
                prompt_used=e_prompt, model_response=img_path or "",
                status="success" if img_path else "error: no image produced",
                task_id=task_id, model_name=model_name,
                operation_type="editing", operation_index=1,
            ))

            done_marker.write_text("ok", encoding="utf-8")
            n_success += 1
        except Exception as exc:
            n_error += 1
            print(f"[{task_id}] case {idx} failed: {exc}")
            traceback.print_exc(limit=2)
            if _is_cuda_fatal(exc):
                break

    writer.close()
    summary = {"task": task_id, "total": len(cases), "success": n_success, "error": n_error, "skipped": n_skipped}
    print(f"[{task_id}] {summary}")
    return summary


# =============================================================================
# TASK: GGU (Generation-Guided Understanding)
# =============================================================================

def run_task_ggu(
    pipeline: InferencePipeline,
    backbone: str,
    data_root: Path,
    out_root: Path,
    max_samples: int,
    resume: bool,
    request_params: Dict[str, Any],
    model_name: str,
) -> Dict[str, Any]:
    task_id = "GGU"
    out_dir = out_root / task_id
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{task_id}_{model_name}_results.csv"
    writer = ResultWriter(csv_path)

    cases = _load_ggu_task_data(data_root / "Gen_Guided_Und")
    if max_samples > 0:
        cases = cases[:max_samples]

    n_success = n_error = n_skipped = 0
    for case in tqdm(cases, desc=f"[{task_id}]"):
        idx = case["index"]
        item_dir = out_dir / f"case_{idx:05d}"
        item_dir.mkdir(parents=True, exist_ok=True)
        done_marker = item_dir / "_done.ok"
        if resume and done_marker.exists():
            n_skipped += 1
            continue

        try:
            has_source_image = bool(case["image_path"]) and Path(case["image_path"]).exists()

            # Understanding (direct): answer the question with only what's given.
            u_prompt = _ggu_understanding_prompt(case["question"], case["options"])
            u_ctx: List[ContextItem] = []
            if has_source_image:
                add_image_path(u_ctx, case["image_path"])
            add_text(u_ctx, u_prompt)
            direct_answer = generate_text_from_context(pipeline, backbone, u_ctx, params=request_params)
            writer.write(_base_row(
                dialogue_index=idx, image_path=case["image_path"], input_data=case,
                prompt_used=u_prompt, model_response=direct_answer or "",
                status="success", task_id=task_id, model_name=model_name,
                operation_type="understanding", operation_index=0,
            ))

            # Generation: illustrate the scene described by `description`.
            g_ctx: List[ContextItem] = []
            add_text(g_ctx, case["description"])
            img_out = item_dir / "model_image.png"
            img_path, _ = generate_image_from_context(
                pipeline, backbone, g_ctx, out_path=img_out, params=request_params,
            )
            writer.write(_base_row(
                dialogue_index=idx, image_path=case["image_path"], input_data=case,
                prompt_used=case["description"], model_response=img_path or "",
                status="success" if img_path else "error: no image produced",
                task_id=task_id, model_name=model_name,
                operation_type="generation", operation_index=1,
            ))

            # Unify: re-answer the question grounded in original + generated image.
            unify_prompt = _ggu_unify_prompt(case["question"], case["options"])
            unify_ctx: List[ContextItem] = []
            if has_source_image:
                add_image_path(unify_ctx, case["image_path"])
            if img_path:
                add_image_path(unify_ctx, img_path)
            add_text(unify_ctx, unify_prompt)
            guided_answer = generate_text_from_context(pipeline, backbone, unify_ctx, params=request_params)
            writer.write(_base_row(
                dialogue_index=idx, image_path=case["image_path"], input_data=case,
                prompt_used=unify_prompt, model_response=guided_answer or "",
                status="success", task_id=task_id, model_name=model_name,
                operation_type="unify", operation_index=2,
            ))

            done_marker.write_text("ok", encoding="utf-8")
            n_success += 1
        except Exception as exc:
            n_error += 1
            print(f"[{task_id}] case {idx} failed: {exc}")
            traceback.print_exc(limit=2)
            if _is_cuda_fatal(exc):
                break

    writer.close()
    summary = {"task": task_id, "total": len(cases), "success": n_success, "error": n_error, "skipped": n_skipped}
    print(f"[{task_id}] {summary}")
    return summary


# =============================================================================
# TASK: ME (Mutual Enhancement)
# =============================================================================

def _run_me_u2g_dialogue(
    pipeline: InferencePipeline,
    backbone: str,
    original_image: str,
    instruction: str,
    max_iterations: int,
    request_params: Dict[str, Any],
    item_dir: Path,
) -> List[Tuple[int, str, str]]:
    """Iterative self-refining editing. Returns [(round_number, op_type, response), ...]."""
    rows: List[Tuple[int, str, str]] = []
    next_instruction = instruction
    current_image = original_image

    for i in range(max_iterations):
        round_num = i * 2 + 1
        edit_ctx: List[ContextItem] = []
        add_image_path(edit_ctx, current_image)
        add_text(edit_ctx, next_instruction if i > 0 else instruction)
        img_out = item_dir / f"u2g_round_{round_num:02d}.png"
        img_path, _ = generate_image_from_context(
            pipeline, backbone, edit_ctx, out_path=img_out, params=request_params,
        )
        rows.append((round_num, "editing", img_path or ""))
        if not img_path:
            break
        current_image = img_path

        round_num += 1
        eval_ctx: List[ContextItem] = []
        add_text(eval_ctx, "Original image:")
        add_image_path(eval_ctx, original_image)
        add_text(eval_ctx, "Edited image:")
        add_image_path(eval_ctx, current_image)
        add_text(eval_ctx, _ME_U2G_EVAL_TEMPLATE.format(instruction=instruction))
        response = generate_text_from_context(pipeline, backbone, eval_ctx, params=request_params)
        rows.append((round_num, "understanding", response or ""))

        should_stop, extracted, format_valid = parse_evaluation_output(response or "")
        if should_stop:
            break
        if format_valid and extracted:
            next_instruction = extracted
        elif not format_valid and (response or "").strip():
            next_instruction = response.strip()
        # else: keep the previous next_instruction as a fallback (or the
        # original instruction on the very first iteration).

    return rows


def _run_me_g2u_dialogue(
    pipeline: InferencePipeline,
    backbone: str,
    original_image: str,
    final_caption: str,
    max_iterations: int,
    request_params: Dict[str, Any],
    item_dir: Path,
) -> List[Tuple[int, str, str]]:
    """Iterative self-refining understanding. Returns [(round_number, op_type, response), ...]."""
    rows: List[Tuple[int, str, str]] = []
    current_image = original_image

    for i in range(max_iterations):
        round_num = i * 2 + 1
        u_ctx: List[ContextItem] = []
        add_image_path(u_ctx, current_image)
        add_text(u_ctx, _ME_G2U_UNDERSTANDING_TEMPLATE.format(caption=final_caption))
        response = generate_text_from_context(pipeline, backbone, u_ctx, params=request_params)
        rows.append((round_num, "understanding", response or ""))

        should_stop, extracted, format_valid = parse_evaluation_output(response or "")
        if should_stop:
            break
        # Mirror upstream's g2u termination rule: a "No" with no extractable
        # instruction, or a totally empty judge response, ends the dialogue
        # instead of feeding a garbage instruction into the next edit.
        if (format_valid and not extracted) or not (response or "").strip():
            break

        combined_instructions = extract_combined_instructions(response or "")
        round_num += 1
        e_ctx: List[ContextItem] = []
        add_image_path(e_ctx, current_image)
        add_text(e_ctx, _me_g2u_editing_prompt(combined_instructions))
        img_out = item_dir / f"g2u_round_{round_num:02d}.png"
        img_path, _ = generate_image_from_context(
            pipeline, backbone, e_ctx, out_path=img_out, params=request_params,
        )
        rows.append((round_num, "editing", img_path or ""))
        if not img_path:
            break
        current_image = img_path

    return rows


def run_task_me(
    pipeline: InferencePipeline,
    backbone: str,
    data_root: Path,
    out_root: Path,
    max_samples: int,
    resume: bool,
    max_iterations: int,
    request_params: Dict[str, Any],
    model_name: str,
) -> Dict[str, Any]:
    task_id = "ME"
    out_dir = out_root / task_id
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{task_id}_{model_name}_results.csv"
    writer = ResultWriter(csv_path)

    cases = _load_me_task_data(data_root / "Mutual_Enhancement")
    if max_samples > 0:
        cases = cases[:max_samples]

    n_success = n_error = n_skipped = 0
    for case in tqdm(cases, desc=f"[{task_id}]"):
        idx = case["index"]
        item_dir = out_dir / f"case_{idx:05d}"
        item_dir.mkdir(parents=True, exist_ok=True)
        done_marker = item_dir / "_done.ok"
        if resume and done_marker.exists():
            n_skipped += 1
            continue

        try:
            u2g_dir = item_dir / "u2g"
            u2g_dir.mkdir(parents=True, exist_ok=True)
            u2g_rows = _run_me_u2g_dialogue(
                pipeline, backbone, case["image_path"], case["instruction"],
                max_iterations, request_params, u2g_dir,
            )
            for round_num, op_type, response in u2g_rows:
                writer.write(_base_row(
                    dialogue_index=idx, image_path=case["image_path"], input_data=case,
                    prompt_used="", model_response=response,
                    status="success", task_id=task_id, model_name=model_name,
                    operation_type=op_type, operation_index=0, round_number=round_num,
                ))

            g2u_dir = item_dir / "g2u"
            g2u_dir.mkdir(parents=True, exist_ok=True)
            g2u_rows = _run_me_g2u_dialogue(
                pipeline, backbone, case["image_path"], case["final_caption"],
                max_iterations, request_params, g2u_dir,
            )
            for round_num, op_type, response in g2u_rows:
                writer.write(_base_row(
                    dialogue_index=idx, image_path=case["image_path"], input_data=case,
                    prompt_used="", model_response=response,
                    status="success", task_id=task_id, model_name=model_name,
                    operation_type=op_type, operation_index=1, round_number=round_num,
                ))

            done_marker.write_text("ok", encoding="utf-8")
            n_success += 1
        except Exception as exc:
            n_error += 1
            print(f"[{task_id}] case {idx} failed: {exc}")
            traceback.print_exc(limit=2)
            if _is_cuda_fatal(exc):
                break

    writer.close()
    summary = {"task": task_id, "total": len(cases), "success": n_success, "error": n_error, "skipped": n_skipped}
    print(f"[{task_id}] {summary}")
    return summary


# =============================================================================
# Main entry point
# =============================================================================

VALID_TASKS = ["IC", "UGG", "GGU", "ME"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Unison generation phase (inference only).")
    parser.add_argument("--config", type=str, required=True, help="Path to a UMM YAML evaluation config file.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    eval_cfg, unison_cfg, inference_cfg = _load_eval_cfg(str(args.config))

    backbone_raw = inference_cfg.get("backbone")
    if not isinstance(backbone_raw, str) or not backbone_raw:
        raise ValueError("`inference.backbone` is required for Unison eval.")
    backbone = _normalize_backbone_name(backbone_raw)

    backbone_cfg = inference_cfg.get("backbone_cfg", {})
    if not isinstance(backbone_cfg, dict):
        raise ValueError("`inference.backbone_cfg` must be a dict when provided.")

    request_cfg = inference_cfg.get("request", {})
    request_params: Dict[str, Any] = {}
    if isinstance(request_cfg, dict):
        params = request_cfg.get("params", {})
        if isinstance(params, dict):
            request_params = dict(params)

    data_root_value = unison_cfg.get("data_root")
    if not data_root_value:
        raise ValueError("`unison.data_root` is required (path to the Unison-Bench `data/` directory).")
    data_root = _resolve_path(str(data_root_value), repo_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Unison data root not found: {data_root}")

    tasks_value = unison_cfg.get("tasks", "all")
    if tasks_value in ("all", None):
        tasks = list(VALID_TASKS)
    elif isinstance(tasks_value, str):
        tasks = [t.strip().upper() for t in tasks_value.split(",") if t.strip()]
    elif isinstance(tasks_value, list):
        tasks = [str(t).strip().upper() for t in tasks_value if str(t).strip()]
    else:
        tasks = list(VALID_TASKS)

    invalid = [t for t in tasks if t not in VALID_TASKS]
    if invalid:
        raise ValueError(f"Unknown Unison tasks: {invalid}. Valid tasks: {VALID_TASKS}")

    out_dir = _resolve_path(str(unison_cfg.get("out_dir", f"output/unison/{backbone}")), repo_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    max_samples = int(unison_cfg.get("max_samples", 0) or 0)
    resume = bool(unison_cfg.get("resume", True))
    max_iterations = max(1, min(5, int(unison_cfg.get("max_iterations", 5) or 5)))

    print(f"[unison generation] backbone={backbone}, tasks={tasks}, data_root={data_root}, out_dir={out_dir}")

    pipeline = InferencePipeline(backbone_name=backbone, backbone_cfg=backbone_cfg)

    task_runners = {
        "IC": lambda: run_task_ic(pipeline, backbone, data_root, out_dir, max_samples, resume, request_params, backbone),
        "UGG": lambda: run_task_ugg(pipeline, backbone, data_root, out_dir, max_samples, resume, request_params, backbone),
        "GGU": lambda: run_task_ggu(pipeline, backbone, data_root, out_dir, max_samples, resume, request_params, backbone),
        "ME": lambda: run_task_me(pipeline, backbone, data_root, out_dir, max_samples, resume, max_iterations, request_params, backbone),
    }

    overall_summary: Dict[str, Any] = {
        "benchmark": "unison",
        "backbone": backbone,
        "out_dir": str(out_dir),
        "tasks": tasks,
        "mode": "generate",
        "task_summaries": {},
    }

    for task_name in tasks:
        try:
            overall_summary["task_summaries"][task_name] = task_runners[task_name]()
        except Exception as exc:
            print(f"[unison generation] Task '{task_name}' failed: {exc}")
            traceback.print_exc()
            overall_summary["task_summaries"][task_name] = {"error": str(exc)}

    del pipeline
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except RuntimeError:
        pass

    summary_path = out_dir / "overall_summary.json"
    summary_path.write_text(json.dumps(overall_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[unison generation] wrote overall summary to {summary_path}")
    print(f"[unison generation] completed. backbone={backbone}, outputs={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
