# MLX Finetuner

Native SwiftUI macOS app for local fine-tuning of Hugging Face models on Apple
Silicon with MLX.

## Architecture

The app uses a SwiftUI frontend and a Python backend process. SwiftUI gives the
workflow a native macOS shell, responsive controls, file pickers, and live
charts. Python owns model inspection, dataset normalization, image validation,
and launching `mlx-lm` or `mlx-vlm`, which keeps MLX integration close to the
official packages.

## Requirements

- Apple Silicon Mac
- macOS 14+
- Xcode command line tools
- Python 3.10+ recommended

The app bootstraps its own Python environment on first launch at:

`~/Library/Application Support/MLXFinetuner/PythonEnv`

It installs the bundled `requirements.txt` there and reuses that environment on
later launches. If you prefer to provide your own environment while developing,
you can still set `MLX_FINETUNER_PYTHON` before running the app or set a Python
path in Settings.

Manual development setup is optional:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Run the app:

```bash
MLX_FINETUNER_PYTHON="$PWD/.venv/bin/python" ./script/build_and_run.sh
```

The Codex Run action is wired to the same script. You can also set the Python
path in the app Settings window.

## Workflow

1. Paste a Hugging Face model ID or local model path.
2. Inspect or download it into a chosen directory.
3. Choose a local `.jsonl` file or folder.
4. Preview normalized samples and image references.
5. Choose `QLoRA` or `Full-parameter` fine-tuning.
6. Configure hyperparameters.
7. Start training and watch loss, throughput, ETA, and logs.
8. Export adapters/full outputs locally or push to Hugging Face.

## Backends

- Text LLMs route to `mlx-lm`.
- Vision-language models route to `mlx-vlm`.
- Detection is based on `config.json` fields such as `model_type`,
  `architectures`, `vision_config`, and known VLM model family hints.

For text QLoRA, the backend quantizes an unquantized base with `mlx_lm.convert`
before launching `mlx_lm.lora`. If the model path already looks quantized, it is
used directly.

## Datasets

Text rows can use:

- `messages`
- `prompt` and `completion`
- `instruction`, optional `input`, and `output`
- `text`

VLM rows should use chat messages with image blocks, or an `images`/`image`
column alongside `messages`. Local paths are resolved relative to the `.jsonl`
file. URL images are checked during preview.

## Example Proofs

See [docs/example-runs.md](docs/example-runs.md) for the two included tiny runs:

- `Examples/text-small`
- `Examples/vlm-small`

Both can be run in dry-run mode immediately. Disable dry-run after installing
MLX dependencies and choosing models that fit your Mac.

## Graceful Failure

Training is a cancellable subprocess. The backend handles Ctrl+C from Terminal
and the app Stop button. If MLX exits with memory or thermal-pressure symptoms,
the UI surfaces a clear suggestion to reduce batch size, sequence length,
gradient accumulation, image resolution, or to use QLoRA instead of full
fine-tuning.
