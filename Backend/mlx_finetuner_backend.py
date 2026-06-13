#!/usr/bin/env python3
"""Backend command surface for MLX Finetuner.

The SwiftUI app calls this script as a subprocess. All command responses are
JSON; training streams newline-delimited JSON events so the UI can stay live.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from dataclasses import asdict, dataclass
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
    parameter_estimate = estimate_parameters(config)

    emit(
        {
            "modelId": model_id,
            "localPath": str(local_path) if local_path else None,
            "kind": kind,
            "modelType": config.get("model_type"),
            "architecture": arch,
            "parameterEstimate": parameter_estimate,
            "warnings": warnings,
        }
    )


def estimate_parameters(config: dict[str, Any]) -> str | None:
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

    for file, line_number, row, error in iter_jsonl(files):
        total += 1
        row_issues: list[str] = []
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
            normalized = normalize_row(row, file.parent)
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


def normalize_row(row: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    issues: list[str] = []
    image_refs: list[ImageRef] = []
    if isinstance(row.get("messages"), list):
        image_refs = extract_images(row, base_dir)
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


def extract_images(row: dict[str, Any], base_dir: Path) -> list[ImageRef]:
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
    return [validate_image_candidate(candidate, base_dir) for candidate in candidates if candidate]


def validate_image_candidate(candidate: Any, base_dir: Path) -> ImageRef:
    if isinstance(candidate, dict):
        candidate = candidate.get("url") or candidate.get("path") or candidate.get("image")
    text = str(candidate)
    if text.startswith(("http://", "https://")):
        return validate_remote_image(text)
    image_path = Path(text).expanduser()
    if not image_path.is_absolute():
        image_path = base_dir / image_path
    ref = ImageRef(path=str(image_path), exists=image_path.exists())
    if not image_path.exists():
        ref.error = "File not found"
        return ref
    enrich_local_image(ref, image_path)
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
        run_training(config)
    except KeyboardInterrupt:
        emit_event("log", level="warning", message="Interrupted. Cleaning up training process.")
        raise SystemExit(130)


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
        for line in process.stdout:
            line = line.rstrip()
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
        suggestion = oom_suggestion()
        emit_event("error", message=f"Training exited with code {code}. {suggestion}")
        raise SystemExit(code)
    emit_event("complete", message="Training complete.", outputPath=str(output_dir))
    if config.get("pushToHF") and config.get("hfRepoId"):
        push_path(str(output_dir), str(config["hfRepoId"]))


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
    command = [
        "mlx_lm.convert",
        "--hf-path",
        model,
        "--mlx-path",
        str(quantized),
        "--quantize",
        "--q-bits",
        str(int(config.get("qloraBits") or 4)),
    ]
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
    assert process.stdout is not None
    for line in process.stdout:
        emit_event("log", message=line.rstrip())
    code = process.wait()
    if code != 0:
        raise RuntimeError(f"Command failed with code {code}: {' '.join(command)}")


def prepare_dataset_for_training(config: dict[str, Any]) -> Path:
    source = Path(config["datasetPath"]).expanduser()
    files = dataset_files(source)
    if not files:
        die(f"No .jsonl files found in {source}")
    work = Path(tempfile.mkdtemp(prefix="mlx-finetuner-data-"))
    train_file = work / "train.jsonl"
    backend = config.get("backend")
    with train_file.open("w", encoding="utf-8") as out:
        for file, _line_number, row, error in iter_jsonl(files):
            if error or row is None:
                continue
            converted = convert_training_row(row, file.parent, backend)
            if converted:
                out.write(json.dumps(converted, ensure_ascii=False) + "\n")
    return work


def convert_training_row(row: dict[str, Any], base_dir: Path, backend: str) -> dict[str, Any] | None:
    if isinstance(row.get("messages"), list):
        if backend == "vision_language":
            converted = dict(row)
            refs = extract_images(row, base_dir)
            if refs:
                converted["images"] = [ref.url or ref.path for ref in refs if ref.exists]
            return converted
        return {"messages": row["messages"]}
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


def build_training_command(
    config: dict[str, Any],
    dataset_dir: Path,
    output_dir: Path,
    model: str,
) -> list[str]:
    if config.get("backend") == "vision_language":
        runner_config = output_dir / "vlm_run_config.json"
        runner_config.write_text(
            json.dumps(
                {
                    "config": config,
                    "dataset": str(dataset_dir / "train.jsonl"),
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
        "adapter_path": str(output_dir / "adapters"),
        "lora_parameters": {
            "rank": int(config.get("loraRank") or 8),
            "dropout": float(config.get("loraDropout") or 0.0),
            "scale": float(config.get("loraAlpha") or 16),
        },
    }
    if config.get("resumeAdapterPath"):
        mlx_payload["resume_adapter_file"] = str(Path(config["resumeAdapterPath"]).expanduser())
    mlx_config.write_text(json.dumps(mlx_payload, indent=2), encoding="utf-8")
    command = [
        "mlx_lm.lora",
        "--config",
        str(mlx_config),
    ]
    return command


def run_vlm_lora(args: argparse.Namespace) -> None:
    payload = load_json(Path(args.config))
    config = payload["config"]
    try:
        import argparse as argparse_module
        from datasets import Dataset
        import mlx_vlm.lora as vlm_lora
    except ImportError as exc:
        raise SystemExit(f"Install VLM dependencies first: pip install mlx-vlm datasets pillow ({exc})")

    dataset_path = payload["dataset"]

    def load_local_or_hub(path: str, dataset_config: str | None = None, split: str = "train", **_kwargs: Any):
        local = Path(path).expanduser()
        if local.exists() and local.suffix == ".jsonl":
            return Dataset.from_json(str(local))
        from datasets import load_dataset

        return load_dataset(path, dataset_config if dataset_config else None, split=split)

    vlm_lora.load_dataset = load_local_or_hub
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
        val_batches=4,
        max_seq_length=int(config["maxSeqLength"]),
        grad_checkpoint=True,
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


def oom_suggestion() -> str:
    return (
        "If this was an out-of-memory or thermal-pressure failure, lower batch size, "
        "max sequence length, gradient accumulation, image resolution, or max pixels; "
        "use QLoRA instead of full fine-tuning for large models."
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
