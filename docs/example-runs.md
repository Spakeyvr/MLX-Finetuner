# Example Runs

These examples are intentionally tiny so they prove the app pipeline without
pretending to produce a useful model.

## Text LLM Dry Run

1. Model: `Qwen/Qwen2.5-0.5B`
2. Dataset: `Examples/text-small`
3. Method: `QLoRA`
4. Enable `Dry run`
5. Start training

The backend normalizes `messages`, `prompt`/`completion`, and
`instruction`/`input`/`output` rows into an `mlx-lm` training directory and
prints the generated `mlx_lm.lora --config ...` command.

## VLM Dry Run

1. Model: `mlx-community/Qwen2-VL-2B-Instruct-4bit`
2. Dataset: `Examples/vlm-small`
3. Method: `QLoRA`
4. Component: `Language model`
5. Enable `Dry run`
6. Preview the dataset and confirm thumbnails appear
7. Start training

The VLM example includes `sample.ppm`, a tiny local image. Preview validates the
image path and creates a thumbnail when Pillow is installed.

## Real Runs

Disable `Dry run` after installing dependencies. For first real runs, keep batch
size at 1 and use a quantized MLX model for QLoRA. Full fine-tuning is much
heavier; expect to reduce sequence length or image resolution on smaller Macs.
