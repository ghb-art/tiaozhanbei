#!/usr/bin/env python3
"""Download only the three additional OpenCodeInstruct shards needed by P0-B1."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[1]
REPO = "nvidia/OpenCodeInstruct"
SHARDS = (8, 25, 42)
LOCAL = ROOT / "data/datasets/opencodeinstruct"
AUDIT = ROOT / "reports/audit/gate_p0b1_code_download.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    outputs = {}
    for shard in SHARDS:
        filename = f"data/train-{shard:05d}-of-00050.parquet"
        downloaded = Path(
            hf_hub_download(
                repo_id=REPO,
                repo_type="dataset",
                revision="main",
                filename=filename,
                local_dir=LOCAL,
            )
        )
        outputs[downloaded.relative_to(ROOT).as_posix()] = {
            "bytes": downloaded.stat().st_size,
            "sha256": sha256_file(downloaded),
        }
        print(f"Ready {downloaded.relative_to(ROOT)}", flush=True)
    report = {
        "gate": "P0-B1-CODE-SOURCE-DOWNLOAD",
        "check_version": "1.0",
        "created_by": "model_compression/download_p0b1_code.py",
        "created_ts": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "repo": REPO,
        "declared_license": "CC BY 4.0",
        "additional_shards": list(SHARDS),
        "outputs": outputs,
        "errors": [],
    }
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {AUDIT.relative_to(ROOT)} status=passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
