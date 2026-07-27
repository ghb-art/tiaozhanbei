#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "p0a4_remediation.json"
FORBIDDEN_IDENTITY_MARKERS = (
    "gsm8k/test/",
    "cmmlu/test/",
    "humaneval/",
    "official_full",
    "final_test",
    "selection170",
    "smoke96",
)

if str(ROOT / "model_compression") not in sys.path:
    sys.path.insert(0, str(ROOT / "model_compression"))

from generate_teacher_capability_distill import validate_code_row  # noqa: E402


class CodeDataError(RuntimeError):
    pass


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise CodeDataError(f"Missing JSONL: {display_path(path)}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise CodeDataError(f"Expected object at {display_path(path)}:{line_number}")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def group_id(row: dict[str, Any]) -> str:
    return str(row.get("validation_group_id") or row.get("sample_id") or "")


def source_identity(row: dict[str, Any]) -> str:
    return " ".join(
        str(row.get(key, "")).lower()
        for key in ("sample_id", "validation_group_id", "source", "split", "split_role")
    )


def require_train_identity(row: dict[str, Any], source_path: Path) -> None:
    identity = source_identity(row)
    if any(marker in identity for marker in FORBIDDEN_IDENTITY_MARKERS):
        raise CodeDataError(
            f"Forbidden evaluation identity in code source {display_path(source_path)}: "
            f"{row.get('sample_id')}"
        )
    if row.get("used_for_training") is not True:
        raise CodeDataError(
            f"Code training row is not explicitly train-only: {row.get('sample_id')}"
        )


def canonical_row(row: dict[str, Any], namespace: str, used_for_training: bool) -> dict[str, Any]:
    code_eval = row.get("code_eval")
    if not isinstance(code_eval, dict) or not code_eval:
        raise CodeDataError(f"Missing executable code_eval: {row.get('sample_id')}")
    messages = row.get("messages")
    answer = str(row.get("answer", "")).strip()
    if not isinstance(messages, list) or not messages or not answer:
        raise CodeDataError(f"Incomplete executable code row: {row.get('sample_id')}")
    group = group_id(row)
    output = {
        "remediation_version": "p0a4r-1.0",
        "created_by": "model_compression/build_p0a4r_code_data.py",
        "source": f"p0a4r_canonical_{row.get('source', 'code_train')}",
        "source_sample_id": str(row.get("sample_id", "")),
        "dataset_key": "humaneval",
        "sample_id": f"p0a4r/{namespace}/{sha256_text(group)[:20]}",
        "validation_group_id": group,
        "messages": messages,
        "answer": answer,
        "code_eval": code_eval,
        "supervision_type": "canonical_executable_solution",
        "used_for_training": used_for_training,
        "used_for_validation": not used_for_training,
        "used_for_final_test": False,
    }
    output["remediation_row_hash"] = sha256_text(
        json.dumps(output, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )
    return output


def unique_code_rows(path: Path, require_train: bool) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        if row.get("dataset_key") != "humaneval":
            continue
        if require_train:
            require_train_identity(row, path)
        group = group_id(row)
        if not group:
            raise CodeDataError(f"Missing validation group in {display_path(path)}")
        unique.setdefault(group, row)
    return [unique[key] for key in sorted(unique)]


def verify_rows(
    rows: list[dict[str, Any]],
    timeout_sec: float,
    workers: int,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    accepted: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()

    def verify(row: dict[str, Any]) -> tuple[dict[str, Any], bool, str]:
        try:
            passed, detail, _ = validate_code_row(row, str(row["answer"]), timeout_sec)
            return row, passed, detail
        except Exception as exc:
            return row, False, f"{type(exc).__name__}: {exc}"

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(verify, row) for row in rows]
        for future in as_completed(futures):
            row, passed, detail = future.result()
            if passed:
                accepted.append(row)
            else:
                rejected[detail] += 1
    accepted.sort(key=lambda row: group_id(row))
    return accepted, rejected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build unique executable train-only Code remediation and internal validation data."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--extra-source",
        action="append",
        default=[],
        help="Additional standardized train-only JSONL with code_eval; may be repeated.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.workers <= 0:
            raise CodeDataError("--workers must be positive")
        config_path = resolve_path(args.config)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        policy = config["policy"]
        if policy.get("feedback_source") != "train_only_internal_validation":
            raise CodeDataError("Remediation policy must be train-only")
        for key in ("smoke96_item_feedback_used", "selection170_feedback_used", "formal_full_feedback_used"):
            if policy.get(key) is not False:
                raise CodeDataError(f"Forbidden feedback policy enabled: {key}")
        data = config["data"]
        settings = config["code_data"]
        primary_path = resolve_path(data["code_primary_source"])
        validation_source_path = resolve_path(data["code_internal_validation_source"])
        train_path = resolve_path(data["code_train"])
        validation_path = resolve_path(data["code_internal_validation"])
        audit_path = ROOT / "reports" / "audit" / "gate_p0a4r_code_data.json"
        configured_optional = [resolve_path(value) for value in data.get("code_optional_sources", [])]
        cli_optional = [resolve_path(value) for value in args.extra_source]
        optional_paths = list(dict.fromkeys(configured_optional + cli_optional))
        present_optional = [path for path in optional_paths if path.is_file()]
        missing_optional = [path for path in optional_paths if not path.is_file()]
        expected_optional_hashes = {
            resolve_path(path): str(digest)
            for path, digest in data.get("code_optional_source_hashes", {}).items()
        }
        for path in present_optional:
            expected_hash = expected_optional_hashes.get(path)
            if expected_hash and sha256_file(path) != expected_hash:
                raise CodeDataError(
                    f"Optional Code source hash mismatch: {display_path(path)}"
                )

        primary_rows = unique_code_rows(primary_path, require_train=True)
        optional_rows_by_path = {
            path: unique_code_rows(path, require_train=True) for path in present_optional
        }
        validation_rows_raw = unique_code_rows(validation_source_path, require_train=False)
        timeout_sec = float(settings["code_timeout_sec"])
        if args.dry_run:
            verified_primary = primary_rows
            verified_optional_by_path = optional_rows_by_path
            verified_validation = validation_rows_raw
            primary_rejections: Counter[str] = Counter()
            optional_rejections: dict[Path, Counter[str]] = {
                path: Counter() for path in present_optional
            }
            validation_rejections: Counter[str] = Counter()
        else:
            verified_primary, primary_rejections = verify_rows(
                primary_rows, timeout_sec, args.workers
            )
            verified_optional_by_path = {}
            optional_rejections = {}
            for path, rows in optional_rows_by_path.items():
                verified, rejected = verify_rows(rows, timeout_sec, args.workers)
                verified_optional_by_path[path] = verified
                optional_rejections[path] = rejected
            verified_validation, validation_rejections = verify_rows(
                validation_rows_raw, timeout_sec, args.workers
            )

        validation_groups = {group_id(row) for row in verified_validation}
        selected_by_group: dict[str, tuple[dict[str, Any], str]] = {}
        for row in verified_primary:
            if group_id(row) not in validation_groups:
                selected_by_group.setdefault(group_id(row), (row, display_path(primary_path)))
        for path in present_optional:
            for row in verified_optional_by_path[path]:
                if group_id(row) not in validation_groups:
                    selected_by_group.setdefault(group_id(row), (row, display_path(path)))

        primary_groups = {group_id(row) for row in verified_primary}
        primary_selected = [
            value for group, value in selected_by_group.items() if group in primary_groups
        ]
        optional_selected = [
            value for group, value in selected_by_group.items() if group not in primary_groups
        ]
        optional_selected.sort(
            key=lambda value: sha256_text(
                f"{config['seed']}:code-extra:{group_id(value[0])}"
            )
        )
        max_train = int(settings["max_train_groups"])
        selected = primary_selected + optional_selected[: max(0, max_train - len(primary_selected))]
        selected.sort(
            key=lambda value: sha256_text(
                f"{config['seed']}:code-train:{group_id(value[0])}"
            )
        )
        train_rows = [
            canonical_row(row, "code-train", used_for_training=True)
            for row, _ in selected
        ]
        internal_validation_rows = [
            canonical_row(row, "internal-code", used_for_training=False)
            for row in verified_validation
        ]
        write_jsonl(train_path, train_rows)
        write_jsonl(validation_path, internal_validation_rows)

        train_groups = {group_id(row) for row in train_rows}
        internal_groups = {group_id(row) for row in internal_validation_rows}
        pilot_min = int(settings["pilot_min_unique_train_groups"])
        promotion_min = int(settings["promotion_min_unique_train_groups"])
        validation_min = int(settings["min_internal_validation_groups"])
        promotion_eligible = (
            len(train_groups) >= promotion_min
            and len(internal_groups) >= validation_min
            and (bool(present_optional) or not settings.get("optional_sources_required_for_promotion", True))
        )
        errors: list[str] = []
        if len(train_groups) < pilot_min:
            errors.append("insufficient_unique_code_groups_for_pilot")
        if len(internal_groups) < validation_min:
            errors.append("insufficient_internal_executable_validation")
        if train_groups & internal_groups:
            errors.append("train_internal_validation_group_overlap")
        source_counts = Counter(str(row.get("source", "")) for row in train_rows)
        audit = {
            "gate": "P0-A4R-EXECUTABLE-CODE-DATA",
            "check_version": "1.0",
            "created_by": "model_compression/build_p0a4r_code_data.py",
            "created_ts": datetime.now(timezone.utc).isoformat(),
            "status": "dry_run_passed" if args.dry_run and not errors else "passed" if not errors else "failed",
            "training_scope": "promotion" if promotion_eligible else "pilot_only",
            "promotion_eligible": promotion_eligible,
            "policy": policy,
            "primary_source": display_path(primary_path),
            "primary_source_hash": sha256_file(primary_path),
            "primary_unique_group_count": len(primary_rows),
            "primary_verified_group_count": len(verified_primary),
            "primary_rejection_counts": dict(primary_rejections),
            "configured_optional_sources": [display_path(path) for path in optional_paths],
            "present_optional_sources": {
                display_path(path): {
                    "sha256": sha256_file(path),
                    "expected_sha256": expected_optional_hashes.get(path, ""),
                    "unique_group_count": len(optional_rows_by_path[path]),
                    "verified_group_count": len(verified_optional_by_path[path]),
                    "rejection_counts": dict(optional_rejections[path]),
                }
                for path in present_optional
            },
            "missing_optional_sources": [display_path(path) for path in missing_optional],
            "internal_validation_source": display_path(validation_source_path),
            "internal_validation_source_hash": sha256_file(validation_source_path),
            "internal_validation_verified_group_count": len(internal_validation_rows),
            "internal_validation_rejection_counts": dict(validation_rejections),
            "training_unique_group_count": len(train_groups),
            "training_source_counts": dict(source_counts),
            "train_internal_validation_group_overlap_count": len(train_groups & internal_groups),
            "pilot_min_unique_train_groups": pilot_min,
            "promotion_min_unique_train_groups": promotion_min,
            "min_internal_validation_groups": validation_min,
            "train": display_path(train_path),
            "train_hash": sha256_file(train_path),
            "internal_validation": display_path(validation_path),
            "internal_validation_hash": sha256_file(validation_path),
            "formal_test_reference_count": 0,
            "errors": errors,
        }
        audit["report_hash"] = sha256_text(
            json.dumps(audit, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        )
        write_json(audit_path, audit)
        print(
            f"Wrote {display_path(train_path)} unique_groups={len(train_groups)} "
            f"scope={audit['training_scope']}"
        )
        print(
            f"Wrote {display_path(validation_path)} unique_groups={len(internal_validation_rows)}"
        )
        print(f"Wrote {display_path(audit_path)} status={audit['status']}")
        if missing_optional and not promotion_eligible:
            print(
                "Optional executable Code sources are missing; this build is pilot-only until "
                f"at least {promotion_min} unique train groups are present.",
                file=sys.stderr,
            )
        elif missing_optional:
            print(
                "Some configured optional Code sources are missing, but the verified unique-group "
                "promotion threshold is satisfied by the available sources.",
                file=sys.stderr,
            )
        return 0 if not errors else 1
    except (OSError, KeyError, ValueError, json.JSONDecodeError, CodeDataError) as exc:
        print(f"P0-A4R Code data build failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
