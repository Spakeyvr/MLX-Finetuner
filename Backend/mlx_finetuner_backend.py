#!/usr/bin/env python3
"""Backend command surface for MLX Finetuner.

The SwiftUI app calls this script as a subprocess. All command responses are
JSON; training streams newline-delimited JSON events so the UI can stay live.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import random
import re
import runpy
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from dataclasses import asdict, dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Iterable


VLM_HINTS = {
    "vl",
    "vision",
    "llava",
    "pixtral",
    "paligemma",
    "mllama",
    "idefics",
    "minicpmv",
    "florence",
    "molmo",
    "qwen2_vl",
    "qwen3_vl",
}

QWEN_HYBRID_LORA_KEYS = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "linear_attn.in_proj_qkv",
    "linear_attn.in_proj_z",
    "linear_attn.in_proj_b",
    "linear_attn.in_proj_a",
    "linear_attn.out_proj",
]

QWEN_LORA_KEYS = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
]

GENERIC_LORA_ALIASES = {
    "q_proj": "self_attn.q_proj",
    "k_proj": "self_attn.k_proj",
    "v_proj": "self_attn.v_proj",
    "o_proj": "self_attn.o_proj",
    "gate_proj": "mlp.gate_proj",
    "up_proj": "mlp.up_proj",
    "down_proj": "mlp.down_proj",
}

GIB = 1024**3
MIB = 1024**2


@dataclass
class ImageRef:
    path: str | None = None
    url: str | None = None
    exists: bool = False
    width: int | None = None
    height: int | None = None
    thumbnailPath: str | None = None
    error: str | None = None


@dataclass
class PreviewExample:
    index: int
    format: str
    prompt: str | None = None
    completion: str | None = None
    messagesSummary: str | None = None
    imageRefs: list[ImageRef] | None = None
    issues: list[str] | None = None


def emit(obj: dict[str, Any]) -> None:
    print(json.dumps(obj, ensure_ascii=False), flush=True)


def emit_event(event_type: str, **payload: Any) -> None:
    emit({"type": event_type, **payload})


def die(message: str, code: int = 1) -> None:
    emit({"ok": False, "message": message})
    raise SystemExit(code)


def safe_model_dir(root: str, model_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "__", model_id).strip("_")
    return Path(root).expanduser() / safe


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalized_model_ref(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def config_text_parts(config: dict[str, Any], model_id: str) -> list[str]:
    parts = [model_id]
    for key in ("model_type", "architectures"):
        value = config.get(key)
        if isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif value:
            parts.append(str(value))
    for nested_key in ("text_config", "vision_config"):
        nested = config.get(nested_key)
        if isinstance(nested, dict):
            nested_type = nested.get("model_type")
            if nested_type:
                parts.append(str(nested_type))
    return parts


def is_qwen_hybrid_reference(config: dict[str, Any], model_id: str) -> bool:
    return any(
        "qwen35" in normalized_model_ref(part)
        or "qwen36" in normalized_model_ref(part)
        for part in config_text_parts(config, model_id)
    )


def is_qwen_reference(config: dict[str, Any], model_id: str) -> bool:
    return any("qwen" in normalized_model_ref(part) for part in config_text_parts(config, model_id))


def is_qwen_hybrid_moe_reference(config: dict[str, Any], model_id: str) -> bool:
    text = " ".join(part.lower() for part in config_text_parts(config, model_id))
    return is_qwen_hybrid_reference(config, model_id) and ("moe" in text or re.search(r"a\d+b", text) is not None)


def model_family(config: dict[str, Any], model_id: str) -> str | None:
    if is_qwen_hybrid_moe_reference(config, model_id):
        return "qwen3_5_moe"
    if is_qwen_hybrid_reference(config, model_id):
        return "qwen3_5"
    if is_qwen_reference(config, model_id):
        return "qwen"
    return None


def detect_model_kind(config: dict[str, Any], model_id: str) -> tuple[str, list[str]]:
    warnings: list[str] = []
    model_type = str(config.get("model_type") or "").lower()
    architectures = [str(item).lower() for item in config.get("architectures", [])]
    text = " ".join([model_id.lower(), model_type, *architectures])
    has_vision_keys = any(
        key in config
        for key in (
            "vision_config",
            "vision_tower",
            "image_token_index",
            "mm_vision_tower",
            "visual",
            "image_processor",
        )
    )
    has_vlm_hint = any(hint in text for hint in VLM_HINTS)
    if is_qwen_hybrid_reference(config, model_id):
        return "vision_language", warnings
    if has_vision_keys or has_vlm_hint:
        return "vision_language", warnings
    if model_type or architectures:
        return "text_llm", warnings
    warnings.append("Could not identify model family from config.json.")
    return "unknown", warnings


def inspect_model(args: argparse.Namespace) -> None:
    model_id = args.model_id.strip()
    local_path: Path | None = None
    config_path: Path | None = None
    warnings: list[str] = []

    candidate = Path(model_id).expanduser()
    if candidate.exists():
        local_path = candidate
        config_path = candidate / "config.json" if candidate.is_dir() else candidate
    else:
        try:
            from huggingface_hub import hf_hub_download, snapshot_download
        except ImportError:
            die("Install huggingface_hub to inspect or download Hub models: pip install huggingface_hub")

        if args.download:
            local_path = safe_model_dir(args.download_dir, model_id)
            local_path.mkdir(parents=True, exist_ok=True)
            snapshot_download(repo_id=model_id, local_dir=str(local_path), local_dir_use_symlinks=False)
            config_path = local_path / "config.json"
        else:
            downloaded = hf_hub_download(repo_id=model_id, filename="config.json")
            config_path = Path(downloaded)

    if not config_path or not config_path.exists():
        die(f"config.json not found for {model_id}")

    config = load_json(config_path)
    kind, detect_warnings = detect_model_kind(config, model_id)
    warnings.extend(detect_warnings)
    architectures = config.get("architectures") or []
    arch = architectures[0] if architectures else None
    family = model_family(config, model_id)
    recommended_settings = recommended_training_settings(config, model_id, kind)
    if family == "qwen3_5":
        warnings.append("Qwen3.5/3.6 hybrid architecture detected; using VLM-safe defaults.")
    elif family == "qwen3_5_moe":
        warnings.append("Qwen3.5/3.6 MoE detected; QLoRA, gradient checkpointing, and small batches are strongly recommended.")
    parameter_estimate = estimate_parameters(config)

    emit(
        {
            "modelId": model_id,
            "localPath": str(local_path) if local_path else None,
            "kind": kind,
            "family": family,
            "modelType": config.get("model_type"),
            "architecture": arch,
            "parameterEstimate": parameter_estimate,
            "recommendedSettings": recommended_settings,
            "warnings": warnings,
        }
    )


def text_config(config: dict[str, Any]) -> dict[str, Any]:
    nested = config.get("text_config")
    return nested if isinstance(nested, dict) else config


def recommended_training_settings(config: dict[str, Any], model_id: str, kind: str) -> dict[str, Any]:
    if is_qwen_reference(config, model_id) and not is_qwen_hybrid_reference(config, model_id):
        settings: dict[str, Any] = {
            "method": "qlora",
            "batchSize": 1,
            "gradientAccumulation": 1,
            "gradCheckpoint": True,
            "qloraBits": 4,
            "loraLayers": 16,
            "loraRank": 8,
            "loraAlpha": 16,
            "loraDropout": 0.0,
            "targetModules": ",".join(QWEN_LORA_KEYS),
        }
        if kind == "vision_language":
            settings.update(
                {
                    "backend": "vision_language",
                    "vlmComponent": "language_model",
                    "imageResolution": 768,
                    "maxPixels": 1_048_576,
                }
            )
        return settings

    if not is_qwen_hybrid_reference(config, model_id):
        return {}
    settings: dict[str, Any] = {
        "backend": "vision_language" if kind == "vision_language" else kind,
        "method": "qlora",
        "vlmComponent": "language_model",
        "batchSize": 1,
        "gradientAccumulation": 1,
        "gradCheckpoint": True,
        "maxSeqLength": 4096,
        "qloraBits": 4,
        "loraLayers": 16,
        "loraRank": 8,
        "loraAlpha": 16,
        "loraDropout": 0.0,
        "targetModules": ",".join(QWEN_HYBRID_LORA_KEYS),
        "imageResolution": 768,
        "maxPixels": 1_048_576,
    }
    if is_qwen_hybrid_moe_reference(config, model_id):
        settings["loraLayers"] = 8
        settings["maxSeqLength"] = 2048
    return settings


def estimate_parameters(config: dict[str, Any]) -> str | None:
    config = text_config(config)
    hidden = config.get("hidden_size") or config.get("n_embd")
    layers = config.get("num_hidden_layers") or config.get("n_layer")
    vocab = config.get("vocab_size")
    intermediate = config.get("intermediate_size") or (hidden * 4 if hidden else None)
    if not all(isinstance(x, int) for x in (hidden, layers, vocab, intermediate)):
        return None
    rough = vocab * hidden + layers * (4 * hidden * hidden + 3 * hidden * intermediate)
    if rough >= 1_000_000_000:
        return f"{rough / 1_000_000_000:.1f}B params rough"
    return f"{rough / 1_000_000:.0f}M params rough"


def dataset_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    preferred = [path / name for name in ("train.jsonl", "data.jsonl", "dataset.jsonl")]
    found = [item for item in preferred if item.exists()]
    if found:
        return found
    return sorted(path.glob("*.jsonl"))


def iter_jsonl(files: Iterable[Path]) -> Iterable[tuple[Path, int, dict[str, Any] | None, str | None]]:
    for file in files:
        with file.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle):
                raw = line.strip()
                if not raw:
                    continue
                try:
                    yield file, line_number, json.loads(raw), None
                except json.JSONDecodeError as exc:
                    yield file, line_number, None, f"Invalid JSON: {exc.msg}"


def preview_dataset(args: argparse.Namespace) -> None:
    path = Path(args.path).expanduser()
    if not path.exists():
        die(f"Dataset path does not exist: {path}")
    files = dataset_files(path)
    if not files:
        die(f"No .jsonl files found in {path}")

    examples: list[PreviewExample] = []
    issues: list[str] = []
    total = 0
    valid = 0
    malformed = 0
    detected_kind = "text_llm"
    image_cache: dict[tuple[str, str, bool], ImageRef] = {}

    for file, line_number, row, error in iter_jsonl(files):
        total += 1
        row_issues: list[str] = []
        include_image_details = len(examples) < args.limit
        if error or row is None:
            malformed += 1
            row_issues.append(error or "Malformed row.")
            example = PreviewExample(
                index=line_number,
                format="malformed",
                prompt=None,
                completion=None,
                messagesSummary=None,
                imageRefs=[],
                issues=row_issues,
            )
        else:
            normalized = normalize_row(row, file.parent, include_image_details, image_cache)
            row_issues.extend(normalized["issues"])
            if normalized["kind"] == "vision_language":
                detected_kind = "vision_language"
            if row_issues:
                malformed += 1
            else:
                valid += 1
            example = PreviewExample(
                index=line_number,
                format=normalized["format"],
                prompt=normalized.get("prompt"),
                completion=normalized.get("completion"),
                messagesSummary=normalized.get("messagesSummary"),
                imageRefs=normalized["imageRefs"],
                issues=row_issues,
            )
        if len(examples) < args.limit:
            examples.append(example)

    if malformed:
        issues.append(f"{malformed} row(s) need attention before training.")
    emit(
        {
            "path": str(path),
            "kind": detected_kind,
            "totalRows": total,
            "validRows": valid,
            "malformedRows": malformed,
            "examples": [serialize_example(example) for example in examples],
            "issues": issues,
        }
    )


def serialize_example(example: PreviewExample) -> dict[str, Any]:
    data = asdict(example)
    data["imageRefs"] = [asdict(ref) for ref in (example.imageRefs or [])]
    data["issues"] = example.issues or []
    return data


def normalize_row(
    row: dict[str, Any],
    base_dir: Path,
    include_image_details: bool = False,
    image_cache: dict[tuple[str, str, bool], ImageRef] | None = None,
) -> dict[str, Any]:
    issues: list[str] = []
    image_refs: list[ImageRef] = []
    if isinstance(row.get("messages"), list):
        image_refs = extract_images(row, base_dir, include_image_details, image_cache)
        kind = "vision_language" if image_refs else "text_llm"
        for ref in image_refs:
            if not ref.exists:
                issues.append(f"Missing image: {ref.path or ref.url or 'unknown'}")
        return {
            "kind": kind,
            "format": "messages",
            "messagesSummary": summarize_messages(row["messages"]),
            "imageRefs": image_refs,
            "issues": issues,
        }
    if "prompt" in row and "completion" in row:
        return {
            "kind": "text_llm",
            "format": "prompt/completion",
            "prompt": trim_text(row.get("prompt")),
            "completion": trim_text(row.get("completion")),
            "imageRefs": [],
            "issues": issues,
        }
    if "instruction" in row and "output" in row:
        prompt = str(row.get("instruction") or "")
        if row.get("input"):
            prompt += "\n" + str(row["input"])
        return {
            "kind": "text_llm",
            "format": "instruction/input/output",
            "prompt": trim_text(prompt),
            "completion": trim_text(row.get("output")),
            "imageRefs": [],
            "issues": issues,
        }
    if "text" in row:
        return {
            "kind": "text_llm",
            "format": "text",
            "prompt": trim_text(row.get("text")),
            "completion": None,
            "imageRefs": [],
            "issues": issues,
        }
    issues.append("Unsupported row format. Expected messages, prompt/completion, instruction/output, or text.")
    return {
        "kind": "unknown",
        "format": "unknown",
        "imageRefs": [],
        "issues": issues,
    }


def trim_text(value: Any, limit: int = 500) -> str:
    text = "" if value is None else str(value)
    return text if len(text) <= limit else text[: limit - 1] + "..."


def summarize_messages(messages: list[Any]) -> str:
    parts: list[str] = []
    for message in messages[:6]:
        if not isinstance(message, dict):
            parts.append("malformed-message")
            continue
        role = message.get("role", "unknown")
        content = message.get("content", "")
        if isinstance(content, list):
            content_text = " ".join(
                block.get("text") or "<image>"
                for block in content
                if isinstance(block, dict)
            )
        else:
            content_text = str(content)
        parts.append(f"{role}: {trim_text(content_text, 120)}")
    return "\n".join(parts)


def extract_images(
    row: dict[str, Any],
    base_dir: Path,
    include_details: bool = False,
    image_cache: dict[tuple[str, str, bool], ImageRef] | None = None,
) -> list[ImageRef]:
    candidates: list[Any] = []
    if isinstance(row.get("images"), list):
        candidates.extend(row["images"])
    elif row.get("image"):
        candidates.append(row["image"])
    for message in row.get("messages", []):
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") in {"image", "image_url"}:
                    candidates.append(block.get("image") or block.get("path") or block.get("url") or block.get("image_url"))
        if isinstance(message.get("images"), list):
            candidates.extend(message["images"])
    return [
        validate_image_candidate(candidate, base_dir, include_details, image_cache)
        for candidate in candidates
        if candidate
    ]


def validate_image_candidate(
    candidate: Any,
    base_dir: Path,
    include_details: bool = False,
    image_cache: dict[tuple[str, str, bool], ImageRef] | None = None,
) -> ImageRef:
    if isinstance(candidate, dict):
        candidate = candidate.get("url") or candidate.get("path") or candidate.get("image")
    text = str(candidate)
    cache_key = (str(base_dir), text, include_details)
    if image_cache is not None and cache_key in image_cache:
        return image_cache[cache_key]
    if text.startswith(("http://", "https://")):
        ref = validate_remote_image(text)
        if image_cache is not None:
            image_cache[cache_key] = ref
        return ref
    image_path = Path(text).expanduser()
    if not image_path.is_absolute():
        image_path = base_dir / image_path
    ref = ImageRef(path=str(image_path), exists=image_path.exists())
    if not image_path.exists():
        ref.error = "File not found"
        if image_cache is not None:
            image_cache[cache_key] = ref
        return ref
    if include_details:
        enrich_local_image(ref, image_path)
    if image_cache is not None:
        image_cache[cache_key] = ref
    return ref


def validate_remote_image(url: str) -> ImageRef:
    ref = ImageRef(url=url)
    try:
        request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "MLXFinetuner/1.0"})
        with urllib.request.urlopen(request, timeout=5) as response:
            content_type = response.headers.get("content-type", "")
            ref.exists = 200 <= response.status < 400 and content_type.startswith("image/")
            if not ref.exists:
                ref.error = f"Remote URL did not return an image ({response.status})."
    except Exception as exc:  # noqa: BLE001 - surface the user-facing validation issue.
        ref.exists = False
        ref.error = str(exc)
    return ref


def enrich_local_image(ref: ImageRef, image_path: Path) -> None:
    try:
        from PIL import Image
    except ImportError:
        return
    try:
        with Image.open(image_path) as image:
            ref.width, ref.height = image.size
            thumb_dir = Path(tempfile.gettempdir()) / "mlx-finetuner-thumbnails"
            thumb_dir.mkdir(parents=True, exist_ok=True)
            thumb = image.copy()
            thumb.thumbnail((192, 144))
            thumb_path = thumb_dir / f"{image_path.stem}-{abs(hash(str(image_path)))}.png"
            thumb.save(thumb_path)
            ref.thumbnailPath = str(thumb_path)
    except Exception as exc:  # noqa: BLE001
        ref.error = str(exc)


def train(args: argparse.Namespace) -> None:
    config = load_json(Path(args.config))
    try:
        run_training(apply_model_compatibility_defaults(config))
    except KeyboardInterrupt:
        emit_event("log", level="warning", message="Interrupted. Cleaning up training process.")
        raise SystemExit(130)


def is_qwen_hybrid_model_id(model_id: str) -> bool:
    text = normalized_model_ref(model_id)
    return "qwen35" in text or "qwen36" in text


def is_qwen_model_id(model_id: str) -> bool:
    return "qwen" in normalized_model_ref(model_id)


def is_qwen_hybrid_moe_model_id(model_id: str) -> bool:
    text = model_id.lower()
    return is_qwen_hybrid_model_id(model_id) and ("moe" in text or re.search(r"a\d+b", text) is not None)


def apply_model_compatibility_defaults(config: dict[str, Any]) -> dict[str, Any]:
    model_id = str(config.get("modelId") or "")
    if not is_qwen_model_id(model_id):
        return config

    updated = dict(config)
    if is_qwen_hybrid_model_id(model_id) or "vl" in normalized_model_ref(model_id):
        updated["backend"] = "vision_language"
    updated["method"] = updated.get("method") or "qlora"
    updated["vlmComponent"] = updated.get("vlmComponent") or "language_model"
    if int(updated.get("batchSize") or 1) > 1:
        updated["batchSize"] = 1
    if is_qwen_hybrid_model_id(model_id) and int(updated.get("maxSeqLength") or 2048) <= 2048:
        updated["maxSeqLength"] = 2048 if is_qwen_hybrid_moe_model_id(model_id) else 4096
    updated["gradCheckpoint"] = True
    updated["qloraBits"] = int(updated.get("qloraBits") or 4)
    if not updated.get("targetModules") or str(updated.get("targetModules")) == "q_proj,v_proj":
        updated["targetModules"] = ",".join(QWEN_HYBRID_LORA_KEYS if is_qwen_hybrid_model_id(model_id) else QWEN_LORA_KEYS)
    if not updated.get("loraLayers"):
        updated["loraLayers"] = 8 if is_qwen_hybrid_moe_model_id(model_id) else 16
    emit_event(
        "log",
        message="Applied Qwen compatibility defaults: QLoRA-oriented LoRA targets and gradient checkpointing.",
    )
    return updated


def parse_positive_float_env(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def mlx_function(mx: Any, name: str) -> Any | None:
    func = getattr(mx, name, None)
    if callable(func):
        return func
    metal = getattr(mx, "metal", None)
    metal_func = getattr(metal, name, None) if metal is not None else None
    return metal_func if callable(metal_func) else None


def mlx_device_info(mx: Any) -> dict[str, Any]:
    info_func = getattr(mx, "device_info", None)
    if callable(info_func):
        try:
            info = info_func()
        except Exception:
            info = {}
        if isinstance(info, dict) and info:
            return info

    metal = getattr(mx, "metal", None)
    info_func = getattr(metal, "device_info", None) if metal is not None else None
    if not callable(info_func):
        return {}
    try:
        info = info_func()
    except Exception:
        return {}
    return info if isinstance(info, dict) else {}


def recommended_mlx_memory_limit(device_info: dict[str, Any]) -> int | None:
    override_gb = parse_positive_float_env("MLX_FINETUNER_MEMORY_LIMIT_GB")
    if override_gb is not None:
        return int(override_gb * GIB)

    working_set = int(device_info.get("max_recommended_working_set_size") or 0)
    total_memory = int(device_info.get("memory_size") or device_info.get("total_memory") or 0)
    candidates: list[int] = []
    if working_set > 0:
        candidates.append(int(working_set * 0.90))
    if total_memory > 0:
        candidates.append(int(total_memory * 0.80))
    if not candidates:
        return None
    return max(2 * GIB, min(candidates))


def recommended_mlx_cache_limit(memory_limit: int) -> int:
    override_gb = parse_positive_float_env("MLX_FINETUNER_CACHE_LIMIT_GB")
    if override_gb is not None:
        return int(override_gb * GIB)
    override_mb = parse_positive_float_env("MLX_FINETUNER_CACHE_LIMIT_MB")
    if override_mb is not None:
        return int(override_mb * MIB)
    return max(256 * MIB, min(2 * GIB, int(memory_limit * 0.08)))


def format_bytes_gib(value: int) -> str:
    return f"{value / GIB:.1f} GiB"


def configure_mlx_runtime(_config: dict[str, Any], emit_log: bool = True) -> None:
    if os.environ.get("MLX_FINETUNER_DISABLE_MEMORY_GUARD") == "1":
        if emit_log:
            emit_event("log", message="MLX memory guard disabled by MLX_FINETUNER_DISABLE_MEMORY_GUARD=1.")
        return

    try:
        import mlx.core as mx
    except ImportError:
        if emit_log:
            emit_event("log", level="warning", message="MLX memory guard skipped because mlx.core is not installed.")
        return

    set_memory_limit = mlx_function(mx, "set_memory_limit")
    set_cache_limit = mlx_function(mx, "set_cache_limit")
    clear_cache = mlx_function(mx, "clear_cache")
    reset_peak_memory = mlx_function(mx, "reset_peak_memory")
    if not callable(set_memory_limit):
        if emit_log:
            emit_event("log", level="warning", message="MLX memory guard skipped because set_memory_limit is unavailable.")
        return

    device_info = mlx_device_info(mx)
    memory_limit = recommended_mlx_memory_limit(device_info)
    if memory_limit is None:
        if emit_log:
            emit_event("log", level="warning", message="MLX memory guard skipped because device memory info is unavailable.")
        return

    try:
        previous_limit = set_memory_limit(memory_limit)
        cache_limit = recommended_mlx_cache_limit(memory_limit)
        if callable(set_cache_limit):
            set_cache_limit(cache_limit)
        if callable(clear_cache):
            clear_cache()
        if callable(reset_peak_memory):
            reset_peak_memory()
    except Exception as exc:
        if emit_log:
            emit_event("log", level="warning", message=f"MLX memory guard could not be applied: {exc}")
        return

    if emit_log:
        details = [
            f"limit {format_bytes_gib(memory_limit)}",
            f"cache {format_bytes_gib(cache_limit)}",
        ]
        if isinstance(previous_limit, int) and previous_limit > 0:
            details.append(f"previous limit {format_bytes_gib(previous_limit)}")
        emit_event("log", message="MLX memory guard active: " + ", ".join(details) + ".")


def write_runtime_config(config: dict[str, Any], output_dir: Path) -> Path:
    path = output_dir / "mlx_runtime_config.json"
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def mlx_module_command(
    module: str,
    module_args: list[str],
    config: dict[str, Any],
    output_dir: Path,
) -> list[str]:
    runtime_config = write_runtime_config(config, output_dir)
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "run-mlx-module",
        "--runtime-config",
        str(runtime_config),
        "--module",
        module,
        "--",
        *module_args,
    ]


def run_training(config: dict[str, Any]) -> None:
    output_dir = Path(config["outputDirectory"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared_dataset = prepare_dataset_for_training(config)
    training_model = prepare_model_for_training(config, output_dir)
    command = build_training_command(config, prepared_dataset, output_dir, training_model)

    if config.get("dryRun"):
        emit_event("log", message="Dry run command:")
        emit_event("log", message=" ".join(command))
        for step in range(1, min(int(config.get("steps") or 8), 8) + 1):
            time.sleep(0.08)
            loss = 2.5 / math.sqrt(step)
            emit_event(
                "metric",
                message=f"dry-run step {step}: loss={loss:.4f}",
                step=step,
                totalSteps=min(int(config.get("steps") or 8), 8),
                loss=loss,
                tokensPerSecond=420.0 + step,
                etaSeconds=max(0, 8 - step) * 0.08,
            )
        emit_event("complete", message="Dry run complete.", outputPath=str(output_dir))
        return

    emit_event("log", message="Starting: " + " ".join(command))
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

    def forward_signal(signum: int, _frame: Any) -> None:
        if process.poll() is None:
            process.send_signal(signum)

    old_int = signal.signal(signal.SIGINT, forward_signal)
    old_term = signal.signal(signal.SIGTERM, forward_signal)
    try:
        assert process.stdout is not None
        suppress_resource_tracker_warning = False
        for line in process.stdout:
            line = line.rstrip()
            if is_resource_tracker_shutdown_warning(line):
                suppress_resource_tracker_warning = True
                continue
            if suppress_resource_tracker_warning:
                if "_recursion_count" in line:
                    suppress_resource_tracker_warning = False
                continue
            if forward_json_event_line(line):
                continue
            parsed = parse_training_line(line)
            if parsed:
                emit_event("metric", message=line, **parsed)
            else:
                emit_event("log", message=line)
        code = process.wait()
    finally:
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)

    if code != 0:
        emit_event("error", message=training_failure_message(code, output_dir))
        raise SystemExit(exit_code_for_child_status(code))
    emit_event("complete", message="Training complete.", outputPath=str(output_dir))
    if config.get("pushToHF") and config.get("hfRepoId"):
        push_path(str(output_dir), str(config["hfRepoId"]))


def is_resource_tracker_shutdown_warning(line: str) -> bool:
    return line.startswith("Exception ignored in: <function ResourceTracker.__del__")


def forward_json_event_line(line: str) -> bool:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return False
    if not isinstance(payload, dict) or "type" not in payload:
        return False
    emit(payload)
    return True


def prepare_model_for_training(config: dict[str, Any], output_dir: Path) -> str:
    model = str(config["modelId"])
    if config.get("backend") != "text_llm" or config.get("method") != "qlora":
        return model
    if is_quantized_model_reference(model):
        return model
    quantized = output_dir / f"quantized-base-{int(config.get('qloraBits') or 4)}bit"
    if config.get("dryRun"):
        emit_event("log", message=f"Would quantize base model to {quantized} before QLoRA.")
        return str(quantized)
    if quantized.exists():
        emit_event("log", message=f"Using existing quantized base at {quantized}.")
        return str(quantized)
    command = mlx_module_command(
        "mlx_lm.convert",
        [
            "--hf-path",
            model,
            "--mlx-path",
            str(quantized),
            "--quantize",
            "--q-bits",
            str(int(config.get("qloraBits") or 4)),
        ],
        config,
        output_dir,
    )
    emit_event("log", message="Quantizing base for QLoRA: " + " ".join(command))
    run_streamed_command(command)
    return str(quantized)


def is_quantized_model_reference(model: str) -> bool:
    text = model.lower()
    return any(hint in text for hint in ("4bit", "8bit", "quant", "q4", "q8"))


def run_streamed_command(command: list[str]) -> None:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    def forward_signal(signum: int, _frame: Any) -> None:
        if process.poll() is None:
            process.send_signal(signum)

    old_int = signal.signal(signal.SIGINT, forward_signal)
    old_term = signal.signal(signal.SIGTERM, forward_signal)
    try:
        assert process.stdout is not None
        for line in process.stdout:
            emit_event("log", message=line.rstrip())
        code = process.wait()
    finally:
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)
    if code != 0:
        raise RuntimeError(f"{training_failure_summary(code)} Command: {' '.join(command)}")


def prepare_dataset_for_training(config: dict[str, Any]) -> Path:
    source = Path(config["datasetPath"]).expanduser()
    files = dataset_files(source)
    if not files:
        die(f"No .jsonl files found in {source}")
    work = Path(tempfile.mkdtemp(prefix="mlx-finetuner-data-"))
    backend = config.get("backend")
    validation_percent = int(config.get("validationSplitPercent") or 0)
    if validation_percent <= 0:
        train_file = work / "train.jsonl"
        count = 0
        image_cache: dict[tuple[str, str, bool], ImageRef] = {}
        with train_file.open("w", encoding="utf-8") as out:
            for file, _line_number, row, error in iter_jsonl(files):
                if error or row is None:
                    continue
                converted = convert_training_row(row, file.parent, backend, image_cache)
                if converted:
                    out.write(json.dumps(converted, ensure_ascii=False) + "\n")
                    count += 1
        if count == 0:
            die("No trainable rows were found after dataset normalization.")
        emit_event("log", message=f"Prepared dataset: {count} train rows, no validation split.")
        return work

    rows: list[dict[str, Any]] = []
    image_cache: dict[tuple[str, str, bool], ImageRef] = {}
    for file, _line_number, row, error in iter_jsonl(files):
        if error or row is None:
            continue
        converted = convert_training_row(row, file.parent, backend, image_cache)
        if converted:
            rows.append(converted)
    if not rows:
        die("No trainable rows were found after dataset normalization.")

    train_rows, valid_rows = split_training_rows(rows, validation_percent)
    write_jsonl(work / "train.jsonl", train_rows)
    if valid_rows:
        write_jsonl(work / "valid.jsonl", valid_rows)
        emit_event("log", message=f"Prepared dataset split: {len(train_rows)} train rows, {len(valid_rows)} validation rows.")
    else:
        emit_event("log", message=f"Prepared dataset: {len(train_rows)} train rows, no validation split.")
    return work


def split_training_rows(rows: list[dict[str, Any]], validation_percent: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    validation_percent = max(0, min(50, validation_percent))
    if validation_percent <= 0 or len(rows) < 2:
        return rows, []
    valid_count = round(len(rows) * validation_percent / 100)
    valid_count = max(1, min(len(rows) - 1, valid_count))
    indices = list(range(len(rows)))
    random.Random(42).shuffle(indices)
    valid_indices = set(indices[:valid_count])
    train_rows = [row for index, row in enumerate(rows) if index not in valid_indices]
    valid_rows = [row for index, row in enumerate(rows) if index in valid_indices]
    return train_rows, valid_rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def convert_training_row(
    row: dict[str, Any],
    base_dir: Path,
    backend: str,
    image_cache: dict[tuple[str, str, bool], ImageRef] | None = None,
) -> dict[str, Any] | None:
    if isinstance(row.get("messages"), list):
        messages = sanitize_messages(row["messages"])
        if not messages:
            return None
        if backend == "vision_language":
            refs = extract_images(row, base_dir, image_cache=image_cache)
            return {
                "messages": messages,
                "images": [ref.url or ref.path for ref in refs if ref.exists],
                "source": str(row.get("source") or ""),
            }
        return {"messages": messages}
    if "prompt" in row and "completion" in row:
        return {"prompt": row["prompt"], "completion": row["completion"]}
    if "instruction" in row and "output" in row:
        prompt = str(row.get("instruction") or "")
        if row.get("input"):
            prompt += "\n" + str(row["input"])
        return {"prompt": prompt, "completion": row["output"]}
    if "text" in row:
        return {"text": row["text"]}
    return None


def sanitize_messages(messages: list[Any]) -> list[dict[str, str]]:
    sanitized: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").strip()
        content = sanitize_message_content(message.get("content"))
        if not role or not content:
            continue
        sanitized.append({"role": role, "content": content})
    return sanitized


def sanitize_message_content(content: Any) -> str:
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            text = block.get("text")
            if text:
                parts.append(str(text))
        return "\n".join(parts).strip()
    if content is None:
        return ""
    return str(content).strip()


def build_training_command(
    config: dict[str, Any],
    dataset_dir: Path,
    output_dir: Path,
    model: str,
) -> list[str]:
    if config.get("backend") == "vision_language":
        runner_config = output_dir / "vlm_run_config.json"
        valid_file = dataset_dir / "valid.jsonl"
        runner_config.write_text(
            json.dumps(
                {
                    "config": config,
                    "dataset": str(dataset_dir / "train.jsonl"),
                    "validDataset": str(valid_file) if valid_file.exists() else None,
                    "model": model,
                    "output": str(output_dir / "adapters.safetensors"),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "run-vlm-lora",
            "--config",
            str(runner_config),
        ]
        return command

    mlx_config = output_dir / "mlx_lm_lora_config.yaml"
    mlx_payload = {
        "model": model,
        "train": True,
        "data": str(dataset_dir),
        "fine_tune_type": "full" if config.get("method") == "full" else "lora",
        "batch_size": int(config["batchSize"]),
        "iters": int(config["steps"]),
        "learning_rate": float(config["learningRate"]),
        "max_seq_length": int(config["maxSeqLength"]),
        "grad_accumulation_steps": int(config["gradientAccumulation"]),
        "num_layers": int(config.get("loraLayers") or 16),
        "grad_checkpoint": bool(config.get("gradCheckpoint", False)),
        "adapter_path": str(output_dir / "adapters"),
        "lora_parameters": {
            "rank": int(config.get("loraRank") or 8),
            "dropout": float(config.get("loraDropout") or 0.0),
            "scale": float(config.get("loraAlpha") or 16),
        },
    }
    lora_keys = lora_target_keys(config)
    if lora_keys:
        mlx_payload["lora_parameters"]["keys"] = lora_keys
    if config.get("resumeAdapterPath"):
        mlx_payload["resume_adapter_file"] = str(Path(config["resumeAdapterPath"]).expanduser())
    mlx_config.write_text(json.dumps(mlx_payload, indent=2), encoding="utf-8")
    return mlx_module_command("mlx_lm.lora", ["--config", str(mlx_config)], config, output_dir)


def lora_target_keys(config: dict[str, Any]) -> list[str]:
    raw = str(config.get("targetModules") or "").strip()
    model_id = str(config.get("modelId") or "")
    if is_qwen_hybrid_model_id(model_id) and (not raw or raw == "q_proj,v_proj"):
        return QWEN_HYBRID_LORA_KEYS
    if is_qwen_model_id(model_id) and (not raw or raw == "q_proj,v_proj"):
        return QWEN_LORA_KEYS

    keys: list[str] = []
    seen: set[str] = set()
    for item in re.split(r"[,;\s]+", raw):
        key = item.strip()
        if not key:
            continue
        key = GENERIC_LORA_ALIASES.get(key, key)
        if key not in seen:
            keys.append(key)
            seen.add(key)
    return keys


def run_vlm_lora(args: argparse.Namespace) -> None:
    payload = load_json(Path(args.config))
    config = payload["config"]
    configure_mlx_runtime(config)
    try:
        import argparse as argparse_module
        from datasets import Dataset, Features, Sequence, Value
        import mlx_vlm.lora as vlm_lora
    except ImportError as exc:
        raise SystemExit(f"Install VLM dependencies first: pip install mlx-vlm datasets pillow ({exc})")

    dataset_path = payload["dataset"]
    if is_qwen_hybrid_model_id(str(config.get("modelId") or payload.get("model") or "")):
        patch_qwen_hybrid_vlm_training(vlm_lora)

    def vlm_json_features() -> Any:
        return Features(
            {
                "messages": [{"role": Value("string"), "content": Value("string")}],
                "images": Sequence(Value("string")),
                "source": Value("string"),
            }
        )

    def load_local_or_hub(path: str, dataset_config: str | None = None, split: str = "train", **_kwargs: Any):
        local = Path(path).expanduser()
        if local.exists() and local.suffix == ".jsonl":
            return Dataset.from_json(str(local), features=vlm_json_features())
        from datasets import load_dataset

        return load_dataset(path, dataset_config if dataset_config else None, split=split)

    vlm_lora.load_dataset = load_local_or_hub
    valid_dataset_path = payload.get("validDataset")
    if valid_dataset_path:
        patch_vlm_validation_dataset(vlm_lora, valid_dataset_path, load_local_or_hub)
    namespace = argparse_module.Namespace(
        model_path=payload["model"],
        full_finetune=config.get("method") == "full",
        train_vision=config.get("vlmComponent") in {"vision_encoder", "both"},
        dataset=dataset_path,
        split="train",
        dataset_config=None,
        image_resize_shape=[int(config["imageResolution"]), int(config["imageResolution"])] if config.get("imageResolution") else None,
        custom_prompt_format=None,
        learning_rate=float(config["learningRate"]),
        batch_size=int(config["batchSize"]),
        iters=int(config["steps"]),
        epochs=int(config["epochs"]) if config.get("epochs") else None,
        steps_per_report=10,
        steps_per_eval=200,
        steps_per_save=100,
        val_batches=int(config.get("vlmValidationBatches") or 1),
        max_seq_length=int(config["maxSeqLength"]),
        grad_checkpoint=bool(config.get("gradCheckpoint", True)),
        grad_clip=None,
        train_on_completions=True,
        gradient_accumulation_steps=int(config["gradientAccumulation"]),
        assistant_id=77091,
        lora_alpha=float(config.get("loraAlpha") or 16),
        lora_rank=int(config.get("loraRank") or 8),
        lora_dropout=float(config.get("loraDropout") or 0.0),
        train_mode="sft",
        beta=0.1,
        eps=1e-8,
        output_path=payload["output"],
        adapter_path=str(Path(config["resumeAdapterPath"]).expanduser()) if config.get("resumeAdapterPath") else None,
    )
    vlm_lora.main(namespace)


def run_mlx_module(args: argparse.Namespace) -> None:
    config = load_json(Path(args.runtime_config))
    configure_mlx_runtime(config)
    module_args = list(args.module_args)
    if module_args and module_args[0] == "--":
        module_args = module_args[1:]
    sys.argv = [args.module, *module_args]
    runpy.run_module(args.module, run_name="__main__", alter_sys=True)


def patch_vlm_validation_dataset(vlm_lora: Any, valid_dataset_path: str, load_dataset_fn: Any) -> None:
    valid_path = Path(valid_dataset_path).expanduser()
    if not valid_path.exists():
        return
    original_train = vlm_lora.train
    train_impl = train_without_initial_validation(original_train)

    @wraps(train_impl)
    def train_with_validation(
        model: Any,
        optimizer: Any,
        train_dataset: Any,
        val_dataset: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if val_dataset is None:
            raw_valid = load_dataset_fn(str(valid_path))
            model_type = getattr(getattr(model, "config", None), "model_type", None)
            raw_valid = vlm_lora.transform_dataset_to_messages(raw_valid, model_type, None)
            val_dataset = train_dataset.__class__(
                raw_valid,
                train_dataset.config,
                train_dataset.processor,
                image_resize_shape=train_dataset.image_resize_shape,
            )
            emit_event("log", message=f"Prepared VLM validation split: {len(raw_valid)} rows.")
        return train_impl(model, optimizer, train_dataset, val_dataset, *args, **kwargs)

    vlm_lora.train = train_with_validation


def train_without_initial_validation(original_train: Any) -> Any:
    try:
        source = inspect.getsource(original_train)
    except Exception:
        return original_train
    old_condition = "it == 1 or it % args.steps_per_eval == 0 or it == args.iters"
    if old_condition not in source:
        return original_train
    source = source.replace(old_condition, "it % args.steps_per_eval == 0 or it == args.iters")
    namespace = dict(original_train.__globals__)
    try:
        exec(source, namespace)
    except Exception:
        return original_train
    patched_train = namespace.get(original_train.__name__)
    return patched_train if callable(patched_train) else original_train


def patch_qwen_hybrid_vlm_training(vlm_lora: Any) -> None:
    emit_event(
        "log",
        message="Applied Qwen3.5/3.6 training patch: disabled gated-delta custom kernels for backward pass.",
    )

    original_setup = vlm_lora.setup_model_for_training

    @wraps(original_setup)
    def setup_model_for_training_with_train_mode(model: Any, args: Any, adapter_path: str | None = None) -> Any:
        patched_model = original_setup(model, args, adapter_path)
        if hasattr(patched_model, "train"):
            patched_model.train()
        return patched_model

    vlm_lora.setup_model_for_training = setup_model_for_training_with_train_mode

    for module_name in (
        "mlx_vlm.models.qwen3_5.gated_delta",
        "mlx_vlm.models.qwen3_5.language",
    ):
        try:
            module = __import__(module_name, fromlist=["dummy"])
        except Exception:
            continue
        for name in (
            "gated_delta_update",
            "gated_delta_update_with_states",
            "gated_delta_state_update",
            "gated_delta_accept_states",
            "_gated_delta_update_verify_decode",
        ):
            original = getattr(module, name, None)
            if original is None or getattr(original, "_mlx_finetuner_safe_kernel", False):
                continue
            parameters = list(inspect.signature(original).parameters)
            use_kernel_index = parameters.index("use_kernel") if "use_kernel" in parameters else None

            @wraps(original)
            def without_custom_kernel(
                *args: Any,
                __original: Any = original,
                __use_kernel_index: int | None = use_kernel_index,
                **kwargs: Any,
            ) -> Any:
                if __use_kernel_index is not None and len(args) > __use_kernel_index:
                    patched_args = list(args)
                    patched_args[__use_kernel_index] = False
                    args = tuple(patched_args)
                    kwargs.pop("use_kernel", None)
                else:
                    kwargs["use_kernel"] = False
                return __original(*args, **kwargs)

            without_custom_kernel._mlx_finetuner_safe_kernel = True
            setattr(module, name, without_custom_kernel)

        if module_name.endswith(".language"):
            patch_qwen_hybrid_language_ops(module)
            patch_qwen_hybrid_rope_ops(module)


def patch_qwen_hybrid_language_ops(language_module: Any) -> None:
    try:
        import mlx.core as mx
        import mlx.nn as nn
    except Exception:
        return

    def precise_swiglu_ops(h: Any, gate: Any, x: Any) -> Any:
        gate = nn.silu(gate.astype(mx.float32))
        x = x.astype(mx.float32)
        return (gate * x).astype(h.dtype)

    def swiglu_ops(gate: Any, x: Any) -> Any:
        return nn.silu(gate) * x

    language_module._precise_swiglu = precise_swiglu_ops
    language_module.swiglu = swiglu_ops

    def target_verify_linears_ops(linears: Any, x: Any, target_verify: bool) -> tuple[Any, ...]:
        return tuple(linear(x) for linear in linears)

    def target_verify_linear_ops(linear: Any, x: Any, target_verify: bool) -> Any:
        return linear(x)

    language_module._target_verify_linears = target_verify_linears_ops
    language_module._target_verify_linear = target_verify_linear_ops

    def no_ragged_decode_attention(*_args: Any, **_kwargs: Any) -> None:
        return None

    language_module._qwen3_5_ragged_decode_attention = no_ragged_decode_attention

    def decode_depthwise_conv_ops(conv_input: Any, weight: Any) -> Any:
        out = mx.sum(conv_input.astype(mx.float32) * weight[None, :, :], axis=1)
        return out.astype(conv_input.dtype)[:, None, :]

    language_module._qwen3_5_decode_depthwise_conv = decode_depthwise_conv_ops


def patch_qwen_hybrid_rope_ops(language_module: Any) -> None:
    try:
        import mlx.core as mx
        import mlx_vlm.models.rope_utils as rope_utils
    except Exception:
        return

    rope_utils._HAS_METAL = False

    def selected_mrope_freqs_ops(position_ids: Any, inv_freq: Any, position_selector: Any) -> Any:
        positions = mx.take(position_ids, position_selector, axis=0).transpose(1, 2, 0)
        return positions.astype(mx.float32) * inv_freq

    def apply_selected_mrope_frequency_layout_ops(freqs: Any, position_selector: Any) -> Any:
        indices = mx.broadcast_to(
            position_selector[None, None, None, :],
            (1, freqs.shape[1], freqs.shape[2], freqs.shape[3]),
        )
        return mx.take_along_axis(freqs, indices, axis=0)[0]

    def apply_interleaved_rotary_axis1_ops(q: Any, k: Any, cos: Any, sin: Any) -> tuple[Any, Any]:
        cos = mx.expand_dims(cos, axis=1)
        sin = mx.expand_dims(sin, axis=1)
        rotary_dim = cos.shape[-1]
        q_rot = q[..., :rotary_dim]
        q_pass = q[..., rotary_dim:]
        k_rot = k[..., :rotary_dim]
        k_pass = k[..., rotary_dim:]
        q_embed = (q_rot * cos) + (rope_utils.rotate_half(q_rot) * sin)
        k_embed = (k_rot * cos) + (rope_utils.rotate_half(k_rot) * sin)
        q_embed = q_embed.astype(q.dtype)
        k_embed = k_embed.astype(k.dtype)
        return (
            mx.concatenate([q_embed, q_pass], axis=-1),
            mx.concatenate([k_embed, k_pass], axis=-1),
        )

    def compute_selected_mrope_cos_sin_ops(
        position_ids: Any,
        inv_freq: Any,
        position_selector: Any,
        frequency_selector: Any,
    ) -> tuple[Any, Any]:
        positions = mx.take(position_ids, position_selector, axis=0).transpose(1, 2, 0)
        freqs = positions.astype(mx.float32) * mx.take(inv_freq, frequency_selector)
        emb = mx.repeat(freqs, repeats=2, axis=-1)
        return mx.cos(emb), mx.sin(emb)

    rope_utils._selected_mrope_freqs = selected_mrope_freqs_ops
    rope_utils._apply_selected_mrope_frequency_layout = apply_selected_mrope_frequency_layout_ops
    rope_utils._apply_interleaved_rotary_pos_emb_axis1 = apply_interleaved_rotary_axis1_ops
    rope_utils.compute_selected_mrope_cos_sin = compute_selected_mrope_cos_sin_ops
    rope_utils._compiled_mrope_apply = lambda *_args, **_kwargs: None
    rope_utils._compiled_rotary_apply = lambda *_args, **_kwargs: None
    rope_utils._maybe_fast_precomputed_rotary = lambda *_args, **_kwargs: None

    language_module._apply_mrope = rope_utils.apply_multimodal_rotary_pos_emb


def parse_training_line(line: str) -> dict[str, Any] | None:
    loss_match = re.search(r"loss[=:\s]+([0-9]+(?:\.[0-9]+)?)", line, re.IGNORECASE)
    step_match = re.search(r"(?:iter|step)[=:\s]+([0-9]+)(?:\s*/\s*([0-9]+))?", line, re.IGNORECASE)
    tok_match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(?:tok/s|tokens/s|tokens/sec)", line, re.IGNORECASE)
    if not (loss_match or step_match or tok_match):
        return None
    payload: dict[str, Any] = {}
    if loss_match:
        payload["loss"] = float(loss_match.group(1))
    if step_match:
        payload["step"] = int(step_match.group(1))
        if step_match.group(2):
            payload["totalSteps"] = int(step_match.group(2))
    if tok_match:
        payload["tokensPerSecond"] = float(tok_match.group(1))
    return payload


def training_failure_summary(code: int) -> str:
    if code < 0:
        signum = -code
        try:
            signal_name = signal.Signals(signum).name
        except ValueError:
            signal_name = f"signal {signum}"
        if signum == signal.SIGABRT:
            return "Training was aborted by SIGABRT while MLX/Metal was running."
        return f"Training was interrupted by {signal_name}."
    return f"Training exited with code {code}."


def latest_adapter_hint(output_dir: Path) -> str | None:
    candidates = [
        output_dir / "adapters.safetensors",
        output_dir / "adapters" / "adapters.safetensors",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def training_failure_message(code: int, output_dir: Path) -> str:
    parts = [training_failure_summary(code)]
    if code < 0 and -code == signal.SIGABRT:
        parts.append("This matches a Metal allocation abort, usually from unified-memory pressure.")
    parts.append(oom_suggestion())
    adapter = latest_adapter_hint(output_dir)
    if adapter:
        parts.append(f"Partial adapters were found at {adapter}; use Resume adapter to continue from them.")
    return " ".join(parts)


def exit_code_for_child_status(code: int) -> int:
    if code < 0:
        return 128 + min(127, -code)
    return code


def oom_suggestion() -> str:
    return (
        "The app applies an MLX memory guard before model loading, but you may still need "
        "to lower batch size, max sequence length, gradient accumulation, image resolution, "
        "or max pixels; use QLoRA instead of full fine-tuning for large models."
    )


def push(args: argparse.Namespace) -> None:
    push_path(args.path, args.repo_id)


def push_path(path: str, repo_id: str) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        die("Install huggingface_hub to push outputs: pip install huggingface_hub")
    api = HfApi()
    target = Path(path).expanduser()
    if target.is_dir():
        api.upload_folder(folder_path=str(target), repo_id=repo_id, repo_type="model")
    else:
        api.upload_file(path_or_fileobj=str(target), path_in_repo=target.name, repo_id=repo_id, repo_type="model")
    emit({"ok": True, "message": f"Pushed {target} to {repo_id}", "outputPath": str(target)})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MLX Finetuner backend")
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect-model")
    inspect.add_argument("--model-id", required=True)
    inspect.add_argument("--download-dir", required=True)
    inspect.add_argument("--download", action="store_true")
    inspect.set_defaults(func=inspect_model)

    preview = sub.add_parser("preview-dataset")
    preview.add_argument("--path", required=True)
    preview.add_argument("--limit", type=int, default=12)
    preview.set_defaults(func=preview_dataset)

    train_parser = sub.add_parser("train")
    train_parser.add_argument("--config", required=True)
    train_parser.set_defaults(func=train)

    vlm_runner = sub.add_parser("run-vlm-lora")
    vlm_runner.add_argument("--config", required=True)
    vlm_runner.set_defaults(func=run_vlm_lora)

    mlx_runner = sub.add_parser("run-mlx-module")
    mlx_runner.add_argument("--runtime-config", required=True)
    mlx_runner.add_argument("--module", required=True)
    mlx_runner.add_argument("module_args", nargs=argparse.REMAINDER)
    mlx_runner.set_defaults(func=run_mlx_module)

    push_parser = sub.add_parser("push")
    push_parser.add_argument("--path", required=True)
    push_parser.add_argument("--repo-id", required=True)
    push_parser.set_defaults(func=push)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except KeyboardInterrupt:
        emit({"ok": False, "message": "Interrupted."})
        raise SystemExit(130)


if __name__ == "__main__":
    main()
