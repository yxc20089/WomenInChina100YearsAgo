#!/usr/bin/env python3
"""Run one official HunyuanOCR task through Transformers on one image."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, HunYuanVLForConditionalGeneration


PROMPTS = {
    "layout": "按照阅读顺序解析图中的版式信息。",
    "spotting_json": (
        "检测并识别图中所有的文字行，请按从上到下、从左到右的阅读顺序进行识别。 "
        "输出格式为 JSON 数组，每个元素必须包含："
        '"box": [xmin, ymin, xmax, ymax]（坐标需归一化到 [0, 1000] 范围内）；'
        '"text": "识别出的文字内容"。 '
        "注意：请直接输出 JSON 数组，不要包含任何多余的描述性文字。"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cuda", "cpu"), default="mps")
    parser.add_argument("--task", choices=tuple(PROMPTS), default="layout")
    parser.add_argument("--prompt", help="experimental prompt override; recorded verbatim")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    image_path = args.image.resolve()
    model_path = args.model.resolve()
    if not image_path.is_file():
        parser.error(f"image does not exist: {image_path}")
    if not model_path.is_dir():
        parser.error(f"model directory does not exist: {model_path}")

    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")

    started_loading = time.perf_counter()
    processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
    model = HunYuanVLForConditionalGeneration.from_pretrained(
        model_path,
        attn_implementation="eager",
        dtype=torch.bfloat16,
    ).to(args.device)
    model.eval()
    loading_seconds = time.perf_counter() - started_loading

    with Image.open(image_path) as raw_image:
        image = raw_image.convert("RGB")
    prompt = args.prompt or PROMPTS[args.task]
    messages = [
        {"role": "system", "content": ""},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=[text],
        images=image,
        padding=True,
        return_tensors="pt",
    ).to(args.device)
    input_ids = inputs["input_ids"] if "input_ids" in inputs else inputs["inputs"]

    started_generation = time.perf_counter()
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            repetition_penalty=1.08,
            use_cache=True,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=(
                processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
            ),
        )
    generation_seconds = time.perf_counter() - started_generation
    trimmed = [
        output[len(prompt_ids) :]
        for prompt_ids, output in zip(input_ids, generated_ids, strict=True)
    ]
    content = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    artifact = {
        "schema_version": "hunyuanocr-transformers-smoke-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "torch_version": torch.__version__,
            "device": args.device,
            "dtype": "bfloat16",
            "attention": "eager",
        },
        "model": {
            "path": str(model_path),
            "model_safetensors_sha256": sha256_file(model_path / "model.safetensors"),
        },
        "request": {
            "task": args.task,
            "prompt": prompt,
            "max_new_tokens": args.max_new_tokens,
            "repetition_penalty": 1.08,
        },
        "image": {
            "path": str(image_path),
            "sha256": sha256_file(image_path),
            "width": image.width,
            "height": image.height,
        },
        "result": {
            "content": content,
            "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "prompt_tokens": int(input_ids.shape[1]),
            "generated_tokens": int(generated_ids.shape[1] - input_ids.shape[1]),
        },
        "timings": {
            "loading_seconds": loading_seconds,
            "generation_seconds": generation_seconds,
        },
    }
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(content)
    print(f"[artifact] {output_path}")
    print(f"[generation] {generation_seconds:.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
