#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/p0a4r2_v1_code.json"


class FreezeError(RuntimeError):
    pass


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FreezeError(f"Missing JSON artifact: {display_path(path)}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FreezeError(f"Expected JSON object: {display_path(path)}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze the selected P0-A4R NLP Adapter lineage.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    return parser.parse_args()


def main() -> int:
    try:
        config_path = resolve_path(parse_args().config)
        config = load_json(config_path)
        frozen = config["frozen_nlp"]
        base = resolve_path(frozen["base_model"])
        base_audit_path = resolve_path(frozen["base_model_audit"])
        train_audit_path = resolve_path(frozen["training_audit"])
        selection_audit_path = resolve_path(frozen["selection_audit"])
        adapter = resolve_path(frozen["adapter"])
        gguf = resolve_path(frozen["gguf_adapter"])
        router_path = resolve_path(frozen["router_manifest"])
        smoke_path = resolve_path(frozen["smoke96_retention_audit"])
        manifest_path = resolve_path(frozen["freeze_manifest"])
        audit_path = resolve_path(frozen["freeze_audit"])

        base_audit = load_json(base_audit_path)
        train_audit = load_json(train_audit_path)
        selection = load_json(selection_audit_path)
        router = load_json(router_path)
        smoke = load_json(smoke_path)
        if not base.is_dir() or base_audit.get("status") != "passed":
            raise FreezeError("Frozen NLP base or base audit is unavailable")
        if resolve_path(str(base_audit.get("output", ""))) != base:
            raise FreezeError("Frozen NLP base audit points to another model")
        if train_audit.get("status") != "passed" or train_audit.get("task") != "cmmlu":
            raise FreezeError("NLP training audit did not pass")
        if resolve_path(str(train_audit.get("model_dir", ""))) != base:
            raise FreezeError("NLP training audit does not use the frozen v2 base")
        if selection.get("status") != "passed" or selection.get("task") != "cmmlu":
            raise FreezeError("NLP checkpoint selection audit did not pass")
        if resolve_path(str(selection.get("selected_output", ""))) != adapter:
            raise FreezeError("NLP selection audit points to another published Adapter")
        if selection.get("selected_checkpoint") not in train_audit.get("checkpoint_candidates", []):
            raise FreezeError("Selected NLP checkpoint is not in the matching training audit")
        if selection.get("selection_data_hash") != train_audit.get("validation_data_hash"):
            raise FreezeError("NLP training and selection validation hashes differ")
        required_files = ("adapter_config.json", "adapter_model.safetensors")
        if not adapter.is_dir() or any(not (adapter / name).is_file() for name in required_files):
            raise FreezeError("Published NLP Adapter is incomplete")
        route = router.get("task_adapters", {}).get("cmmlu", {})
        if not gguf.is_file() or route.get("path") != display_path(gguf):
            raise FreezeError("NLP GGUF Adapter does not match its router entry")
        if route.get("sha256") != sha256_file(gguf):
            raise FreezeError("NLP GGUF Adapter hash changed")
        ratios = smoke.get("ratios", {})
        if smoke.get("generation_error_count") != 0 or float(ratios.get("nlp_ratio", 0)) < 0.75:
            raise FreezeError("NLP smoke96 evidence is not eligible for archival freeze")
        if frozen.get("deployment_on_v1") is not False:
            raise FreezeError("A v2 NLP Adapter must not be enabled on the v1 base")

        adapter_files = {
            name: {
                "sha256": sha256_file(adapter / name),
                "bytes": (adapter / name).stat().st_size,
            }
            for name in required_files
        }
        manifest = {
            "freeze_version": "p0a4r-nlp-v2-1.0",
            "created_by": "scripts/freeze_p0a4r_nlp.py",
            "created_ts": datetime.now(timezone.utc).isoformat(),
            "status": "frozen",
            "task": "cmmlu",
            "base_model": display_path(base),
            "base_model_audit": display_path(base_audit_path),
            "base_model_audit_sha256": sha256_file(base_audit_path),
            "training_audit": display_path(train_audit_path),
            "training_audit_sha256": sha256_file(train_audit_path),
            "selection_audit": display_path(selection_audit_path),
            "selection_audit_sha256": sha256_file(selection_audit_path),
            "selected_checkpoint": selection["selected_checkpoint"],
            "adapter": display_path(adapter),
            "adapter_files": adapter_files,
            "gguf_adapter": display_path(gguf),
            "gguf_adapter_sha256": sha256_file(gguf),
            "smoke96_retention_audit": display_path(smoke_path),
            "smoke96_retention_audit_sha256": sha256_file(smoke_path),
            "smoke96_nlp_ratio": float(ratios["nlp_ratio"]),
            "generation_error_count": int(smoke["generation_error_count"]),
            "deployment_on_v1": False,
            "deployment_policy": "archive_only; v1 uses shared NLP",
        }
        manifest["manifest_hash"] = sha256_json(manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        if manifest_path.exists():
            previous = load_json(manifest_path)
            immutable_keys = (
                "base_model_audit_sha256",
                "training_audit_sha256",
                "selection_audit_sha256",
                "adapter_files",
                "gguf_adapter_sha256",
                "smoke96_retention_audit_sha256",
            )
            changed = [
                key for key in immutable_keys if previous.get(key) != manifest.get(key)
            ]
            if changed:
                raise FreezeError(f"Existing frozen NLP lineage changed: {changed}")
            manifest = previous
        else:
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

        audit = {
            "gate": "P0-A4R-NLP-FREEZE",
            "check_version": "1.0",
            "created_by": "scripts/freeze_p0a4r_nlp.py",
            "created_ts": datetime.now(timezone.utc).isoformat(),
            "status": "passed",
            "manifest": display_path(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "base_compatibility": "v2_only",
            "v1_deployment": "shared_model_no_nlp_adapter",
            "formal_test_reference_count": 0,
        }
        audit["report_hash"] = sha256_json(audit)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        if audit_path.exists():
            previous_audit = load_json(audit_path)
            if (
                previous_audit.get("status") != "passed"
                or previous_audit.get("manifest_sha256") != audit["manifest_sha256"]
                or previous_audit.get("base_compatibility") != "v2_only"
                or previous_audit.get("v1_deployment") != "shared_model_no_nlp_adapter"
            ):
                raise FreezeError("Existing NLP freeze audit does not match the frozen manifest")
        else:
            audit_path.write_text(
                json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        print(f"Frozen NLP lineage: {display_path(manifest_path)}")
        print(f"Audit: {display_path(audit_path)}")
        return 0
    except (OSError, KeyError, ValueError, json.JSONDecodeError, FreezeError) as exc:
        print(f"P0-A4R NLP freeze failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
