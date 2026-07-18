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
DEFAULT_AUDIT_COPY_PATH = ROOT / "reports" / "audit" / "model_downloads.json"

MODEL_SPECS = [
    {
        "role": "cloud_teacher",
        "repo_id": "Qwen/Qwen2.5-14B-Instruct-AWQ",
        "local_name": "Qwen--Qwen2.5-14B-Instruct-AWQ",
    },
    {
        "role": "edge_student",
        "repo_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "local_name": "deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B",
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


def load_existing_summary(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(value, list):
        return []
    rows = [dict(row) for row in value if isinstance(row, dict)]
    return rows


def merge_summary(existing: list[dict[str, Any]], downloaded: list[dict[str, Any]]) -> list[dict[str, Any]]:
    configured_repos = {spec["repo_id"] for spec in MODEL_SPECS}
    merged = {
        str(row.get("repo_id", "")): row
        for row in existing
        if str(row.get("repo_id", "")) in configured_repos
    }
    for row in downloaded:
        merged[str(row["repo_id"])] = row
    role_order = {spec["role"]: index for index, spec in enumerate(MODEL_SPECS)}
    return sorted(
        merged.values(),
        key=lambda row: (role_order.get(str(row.get("role", "")), len(role_order)), str(row.get("repo_id", ""))),
    )


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
    parser.add_argument(
        "--audit-copy",
        default=str(DEFAULT_AUDIT_COPY_PATH),
        help="Second copy of the merged model audit. Use an empty value to disable.",
    )
    parser.add_argument("--max-workers", type=int, default=1, help="Concurrent Hugging Face download workers.")
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Audit already-present configured snapshots without contacting Hugging Face.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    summary_path = Path(args.summary)
    if not summary_path.is_absolute():
        summary_path = ROOT / summary_path

    audit_copy_path = Path(args.audit_copy) if args.audit_copy else None
    if audit_copy_path is not None and not audit_copy_path.is_absolute():
        audit_copy_path = ROOT / audit_copy_path

    try:
        specs = selected_models(args.model)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[dict[str, Any]] = []
    print(f"Model download started at {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    for spec in specs:
        local_dir = output_dir / spec["local_name"]
        if args.local_only:
            if not local_dir.is_dir():
                print(f"Missing configured local snapshot: {local_dir}", file=sys.stderr)
                return 1
            print(f"AUDIT {spec['role']} {spec['repo_id']} -> {local_dir}", flush=True)
        else:
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
            "local_status": "audited_existing" if args.local_only else "downloaded",
            **stats,
        }
        downloaded.append(item)
        print(
            f"DONE {spec['role']} files={stats['file_count']} bytes={stats['bytes']}",
            flush=True,
        )

    summary = merge_summary(load_existing_summary(summary_path), downloaded)
    summary_text = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(summary_text, encoding="utf-8")
    if audit_copy_path is not None:
        audit_copy_path.parent.mkdir(parents=True, exist_ok=True)
        audit_copy_path.write_text(summary_text, encoding="utf-8")
    print(f"SUMMARY {summary_path.relative_to(ROOT).as_posix()}", flush=True)
    if audit_copy_path is not None:
        print(f"AUDIT {audit_copy_path.relative_to(ROOT).as_posix()}", flush=True)
    print(f"Model download finished at {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
