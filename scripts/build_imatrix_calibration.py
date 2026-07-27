from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "distill" / "g0_imatrix_calibration.txt"
DEFAULT_AUDIT = ROOT / "reports" / "audit" / "g0_imatrix_calibration.json"


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def render_row(row: dict[str, Any]) -> str:
    parts: list[str] = []
    for message in row.get("messages", []):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "user"))
        content = str(message.get("content", "")).strip()
        if content:
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    answer = str(row.get("answer") or row.get("chosen") or "").strip()
    if answer:
        parts.append(f"<|im_start|>assistant\n{answer}<|im_end|>")
    return "\n".join(parts)


def formal_test_reference(row: dict[str, Any]) -> bool:
    searchable = " ".join(
        str(row.get(key, "")).lower()
        for key in ("sample_id", "original_sample_id", "split", "split_role", "source_split")
    )
    return any(marker in searchable for marker in ("humaneval/test", "gsm8k/test", "cmmlu/test", "final_test"))


def select_rows(
    rows: list[dict[str, Any]],
    rng: random.Random,
    rows_per_source: int,
    stratify_key: str = "",
    strata: tuple[str, ...] = (),
    rows_per_stratum: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if not stratify_key:
        shuffled = list(rows)
        rng.shuffle(shuffled)
        return shuffled[:rows_per_source], {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        value = str(row.get(stratify_key, "")).strip()
        if value:
            grouped.setdefault(value, []).append(row)
    selected_strata = strata or tuple(sorted(grouped))
    if not selected_strata:
        raise ValueError(f"No non-empty values for stratify key {stratify_key!r}")
    if rows_per_stratum <= 0:
        raise ValueError("--rows-per-stratum must be > 0 when --stratify-key is used")
    selected: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for stratum in selected_strata:
        candidates = list(grouped.get(stratum, []))
        if len(candidates) < rows_per_stratum:
            raise ValueError(
                f"Stratum {stratum!r} has {len(candidates)} rows; "
                f"requires {rows_per_stratum}"
            )
        rng.shuffle(candidates)
        chosen = candidates[:rows_per_stratum]
        selected.extend(chosen)
        counts[stratum] = len(chosen)
    rng.shuffle(selected)
    return selected, counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a train-only mixed calibration corpus for llama.cpp imatrix.")
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--rows-per-source", type=int, default=256)
    parser.add_argument("--stratify-key", default="")
    parser.add_argument("--stratum", action="append", default=[])
    parser.add_argument("--rows-per-stratum", type=int, default=0)
    parser.add_argument("--seed", type=int, default=202606)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.rows_per_source <= 0:
        print("--rows-per-source must be > 0", file=sys.stderr)
        return 2
    sources = [resolve_path(value) for value in args.source]
    output = resolve_path(args.output)
    audit_path = resolve_path(args.audit)
    rng = random.Random(args.seed)
    selected_texts: list[str] = []
    source_counts: dict[str, int] = {}
    stratum_counts_by_source: dict[str, dict[str, int]] = {}
    selected_stratum_counts: dict[str, int] = {}
    formal_test_reference_count = 0
    errors: list[str] = []
    for source in sources:
        if not source.is_file():
            errors.append(f"Missing source: {display_path(source)}")
            continue
        rows = load_jsonl(source)
        try:
            selected, stratum_counts = select_rows(
                rows,
                rng,
                args.rows_per_source,
                args.stratify_key,
                tuple(args.stratum),
                args.rows_per_stratum,
            )
        except ValueError as exc:
            errors.append(f"{display_path(source)}: {exc}")
            continue
        source_counts[display_path(source)] = len(selected)
        if stratum_counts:
            stratum_counts_by_source[display_path(source)] = stratum_counts
            for stratum, count in stratum_counts.items():
                selected_stratum_counts[stratum] = selected_stratum_counts.get(stratum, 0) + count
        for row in selected:
            if formal_test_reference(row):
                formal_test_reference_count += 1
                continue
            text = render_row(row)
            if text:
                selected_texts.append(text)
    if formal_test_reference_count:
        errors.append(f"formal test references detected: {formal_test_reference_count}")
    if not selected_texts:
        errors.append("No calibration texts were selected")
    if not errors:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n\n".join(selected_texts) + "\n", encoding="utf-8")
    status = "passed" if not errors and output.is_file() else "failed"
    audit = {
        "gate": "G0-imatrix-calibration",
        "check_version": "1.0",
        "created_by": "scripts/build_imatrix_calibration.py",
        "created_ts": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "sources": {display_path(path): sha256_file(path) for path in sources if path.is_file()},
        "source_counts": source_counts,
        "stratify_key": args.stratify_key,
        "requested_strata": args.stratum,
        "rows_per_stratum": args.rows_per_stratum,
        "stratum_counts_by_source": stratum_counts_by_source,
        "selected_stratum_counts": selected_stratum_counts,
        "selected_text_count": len(selected_texts),
        "formal_test_reference_count": formal_test_reference_count,
        "seed": args.seed,
        "output": display_path(output),
        "output_hash": sha256_file(output) if output.is_file() else "",
        "output_bytes": output.stat().st_size if output.is_file() else 0,
        "errors": errors,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {display_path(audit_path)}")
    print(f"status={status} selected={len(selected_texts)}")
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
