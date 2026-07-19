#!/usr/bin/env python3
"""Run provenance-capturing HunyuanOCR requests against llama-server."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import subprocess
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "pipeline-models.toml"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_prompts(config_path: Path) -> dict[str, str]:
    with config_path.open("rb") as source:
        config = tomllib.load(source)
    selected = config["ocr"]
    return {
        selected["spotting_task"]: selected["spotting_prompt"],
        selected["layout_task"]: selected["layout_prompt"],
        "structured_parse": "提取图中的文字。",
    }


def image_part(path: Path) -> dict[str, Any]:
    media_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{media_type};base64,{encoded}"},
    }


def call_server(
    endpoint: str,
    model: str,
    image_path: Path,
    prompt: str,
    max_tokens: int,
) -> dict[str, Any]:
    body = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    image_part(image_path),
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1,
        "top_k": 1,
        "repetition_penalty": 1,
        "seed": 42,
        "stream": False,
    }
    request_bytes = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = Request(
        endpoint.rstrip("/") + "/v1/chat/completions",
        data=request_bytes,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=900) as response:
            envelope = json.load(response)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"llama-server returned HTTP {exc.code}: {detail}") from exc
    elapsed = time.perf_counter() - started
    choice = envelope["choices"][0]
    content = choice["message"].get("content") or ""
    return {
        "elapsed_seconds": elapsed,
        "finish_reason": choice.get("finish_reason"),
        "content": content,
        "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "usage": envelope.get("usage"),
        "timings": envelope.get("timings"),
        "response_id": envelope.get("id"),
    }


def executable_version(executable: Path) -> str:
    result = subprocess.run(
        [str(executable), "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    return (result.stdout + result.stderr).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18080")
    parser.add_argument("--model", default="HYVL")
    parser.add_argument(
        "--task",
        choices=("structured_parse", "spotting_json", "layout_parse"),
        default="spotting_json",
    )
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--model-gguf", type=Path, required=True)
    parser.add_argument("--mmproj-gguf", type=Path, required=True)
    parser.add_argument("--llama-server", type=Path, required=True)
    parser.add_argument(
        "--acceleration",
        choices=("metal", "cpu"),
        default="metal",
        help="launch mode to record; the runner cannot inspect server device placement",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    paths = [path.resolve() for path in args.images]
    for path in paths:
        if not path.is_file():
            parser.error(f"image does not exist: {path}")
    prompts = load_prompts(args.config.resolve())
    prompt = prompts[args.task]

    results = []
    for path in paths:
        print(f"[{args.task}] {path.name}", flush=True)
        result = call_server(
            args.endpoint,
            args.model,
            path,
            prompt,
            args.max_tokens,
        )
        print(result["content"], flush=True)
        print(f"[elapsed] {result['elapsed_seconds']:.3f}s", flush=True)
        results.append(
            {
                "image": str(path.relative_to(PROJECT_ROOT)),
                "image_sha256": sha256_file(path),
                "result": result,
            }
        )

    artifact = {
        "schema_version": "hunyuanocr-llamacpp-smoke-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "endpoint": args.endpoint,
            "model_alias": args.model,
            "llama_server_version": executable_version(args.llama_server.resolve()),
            "acceleration_requested": args.acceleration,
            "acceleration_verification": "verify device placement in llama-server log",
        },
        "model": {
            "model_gguf_sha256": sha256_file(args.model_gguf.resolve()),
            "mmproj_gguf_sha256": sha256_file(args.mmproj_gguf.resolve()),
        },
        "request": {
            "task": args.task,
            "prompt": prompt,
            "temperature": 0,
            "top_p": 1,
            "top_k": 1,
            "repetition_penalty": 1,
            "seed": 42,
            "max_tokens": args.max_tokens,
        },
        "results": results,
        "all_requests_stopped": all(
            item["result"]["finish_reason"] == "stop" for item in results
        ),
    }
    if args.output:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[artifact] {output}", flush=True)
    else:
        print(json.dumps(artifact, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
