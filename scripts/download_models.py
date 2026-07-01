from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "models" / "pretrained"
DEFAULT_SUMMARY_PATH = DEFAULT_OUTPUT_DIR / "MODEL_DOWNLOADS.json"

MODEL_SPECS = [
    {
        "role": "cloud_teacher",
        "repo_id": "Qwen/Qwen2.5-14B-Instruct-AWQ",
        "local_name": "Qwen--Qwen2.5-14B-Instruct-AWQ",
    },
    {
        "role": "high_edge",
        "repo_id": "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "local_name": "Qwen--Qwen2.5-7B-Instruct-AWQ",
    },
    {
        "role": "edge_student",
        "repo_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "local_name": "Qwen--Qwen2.5-1.5B-Instruct",
    },
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_stats(path: Path) -> dict[str, Any]:
    files = [
        file_path
        for file_path in sorted(path.rglob("*"))
        if file_path.is_file() and ".cache" not in file_path.relative_to(path).parts
    ]
    return {
        "file_count": len(files),
        "bytes": sum(file_path.stat().st_size for file_path in files),
        "key_files": [
            {
                "path": file_path.relative_to(path).as_posix(),
                "bytes": file_path.stat().st_size,
                "sha256": sha256_file(file_path),
            }
            for file_path in files
            if file_path.name in {
                "config.json",
                "generation_config.json",
                "tokenizer_config.json",
                "model.safetensors.index.json",
            }
        ],
        "weight_files": [
            {
                "path": file_path.relative_to(path).as_posix(),
                "bytes": file_path.stat().st_size,
                "sha256": sha256_file(file_path),
            }
            for file_path in files
            if file_path.suffix == ".safetensors"
        ],
    }


def selected_models(names: list[str]) -> list[dict[str, str]]:
    if not names:
        return MODEL_SPECS

    by_role = {spec["role"]: spec for spec in MODEL_SPECS}
    by_repo = {spec["repo_id"]: spec for spec in MODEL_SPECS}
    selected: list[dict[str, str]] = []
    for name in names:
        spec = by_role.get(name) or by_repo.get(name)
        if spec is None:
            raise ValueError(f"Unknown model selector: {name}")
        selected.append(spec)
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download project model snapshots from Hugging Face.")
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Model role or repo id to download. Can be repeated. Defaults to all configured models.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for local model snapshots.")
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY_PATH), help="Download summary JSON path.")
    parser.add_argument("--max-workers", type=int, default=1, help="Concurrent Hugging Face download workers.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    summary_path = Path(args.summary)
    if not summary_path.is_absolute():
        summary_path = ROOT / summary_path

    try:
        specs = selected_models(args.model)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, Any]] = []
    print(f"Model download started at {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    for spec in specs:
        local_dir = output_dir / spec["local_name"]
        local_dir.mkdir(parents=True, exist_ok=True)
        print(f"START {spec['role']} {spec['repo_id']} -> {local_dir}", flush=True)
        snapshot_download(
            repo_id=spec["repo_id"],
            local_dir=str(local_dir),
            max_workers=args.max_workers,
        )
        stats = directory_stats(local_dir)
        item = {
            "role": spec["role"],
            "repo_id": spec["repo_id"],
            "local_dir": local_dir.relative_to(ROOT).as_posix(),
            **stats,
        }
        summary.append(item)
        print(
            f"DONE {spec['role']} files={stats['file_count']} bytes={stats['bytes']}",
            flush=True,
        )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"SUMMARY {summary_path.relative_to(ROOT).as_posix()}", flush=True)
    print(f"Model download finished at {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
