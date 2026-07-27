#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "model_compression") not in sys.path:
    sys.path.insert(0, str(ROOT / "model_compression"))

from code_contests_utils import (  # noqa: E402
    normalize_tests,
    run_contest_tests,
    safe_contest_source,
    stable_test_hash,
)


PARQUET_API = (
    "https://huggingface.co/api/datasets/"
    "deepmind/code_contests/parquet/default/train"
)
DEFAULT_PARQUET_DIR = (
    ROOT / "data" / "datasets" / "code_contests" / "parquet" / "train"
)
DEFAULT_TRAIN = ROOT / "data" / "distill" / "p0a4r3_code_contests_train.jsonl"
DEFAULT_VALIDATION = (
    ROOT / "data" / "distill" / "p0a4r3_code_contests_validation.jsonl"
)
DEFAULT_CACHE = ROOT / "runtime" / "p0a4r3_code_contests_validation.jsonl"
DEFAULT_AUDIT = ROOT / "reports" / "audit" / "gate_p0a4r3_code_contests.json"
DEFAULT_HUMANEVAL = (
    ROOT / "data" / "datasets" / "humaneval" / "data" / "HumanEval.jsonl.gz"
)
DEFAULT_TOKENIZER = ROOT / "models" / "checkpoints" / "p0a4" / "student-shared-merged"


class CodeContestsError(RuntimeError):
    pass


@dataclass(frozen=True)
class Candidate:
    cache_key: str
    group_id: str
    name: str
    description: str
    origin: str
    difficulty: str
    tests: tuple[dict[str, str], ...]
    solutions: tuple[str, ...]


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
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(stable_json(row) + "\n")


def download_train_parquet(output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with urlopen(PARQUET_API, timeout=30) as response:
        urls = json.loads(response.read().decode("utf-8"))
    if not isinstance(urls, list) or not urls:
        raise CodeContestsError("Hugging Face parquet API returned no train shards")
    outputs = [
        output_dir / f"train-{index:04d}.parquet"
        for index in range(len(urls))
    ]

    def download_one(index: int, url: str) -> Path:
        path = output_dir / f"train-{index:04d}.parquet"
        partial = path.with_suffix(path.suffix + ".part")
        import pyarrow.parquet as pq

        if path.is_file():
            try:
                pq.ParquetFile(path)
                print(f"[CodeContests download] existing {display_path(path)}", flush=True)
                return path
            except Exception:
                if partial.exists():
                    raise CodeContestsError(
                        "Both incomplete final and partial parquet exist: "
                        f"{display_path(path)}, {display_path(partial)}"
                    )
                path.replace(partial)
        command = [
            "curl",
            "--fail",
            "--location",
            "--silent",
            "--show-error",
            "--retry",
            "5",
            "--retry-all-errors",
            "--retry-delay",
            "2",
            "--continue-at",
            "-",
            "--output",
            str(partial),
            str(url),
        ]
        print(
            f"[CodeContests download] {index + 1}/{len(urls)} -> {display_path(path)}",
            flush=True,
        )
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise CodeContestsError(
                f"Download failed for shard {index}: curl={completed.returncode}"
            )
        try:
            pq.ParquetFile(partial)
        except Exception as exc:
            raise CodeContestsError(
                f"Downloaded shard is not valid parquet: {display_path(partial)}"
            ) from exc
        partial.replace(path)
        print(
            f"[CodeContests download] ready {index + 1}/{len(urls)} "
            f"bytes={path.stat().st_size}",
            flush=True,
        )
        return path

    with ThreadPoolExecutor(max_workers=min(4, len(urls))) as executor:
        futures = {
            executor.submit(download_one, index, str(url)): index
            for index, url in enumerate(urls)
        }
        for future in as_completed(futures):
            future.result()
    write_json(
        output_dir.parent / "SOURCE.json",
        {
            "dataset": "deepmind/code_contests",
            "upstream_repository": "https://github.com/google-deepmind/code_contests",
            "distribution": "Hugging Face auto-converted parquet",
            "parquet_api": PARQUET_API,
            "license": "CC-BY-4.0 for non-code materials; upstream component terms apply",
            "split": "train_only",
            "validation_or_test_downloaded": False,
            "shards": [
                {
                    "path": display_path(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in outputs
            ],
            "created_ts": datetime.now(timezone.utc).isoformat(),
        },
    )
    return outputs


def solution_pairs(value: Any) -> list[tuple[int, str]]:
    if isinstance(value, dict):
        languages = value.get("language", [])
        solutions = value.get("solution", [])
        if isinstance(languages, list) and isinstance(solutions, list):
            return [
                (int(language), str(solution))
                for language, solution in zip(languages, solutions)
            ]
    if isinstance(value, list):
        pairs: list[tuple[int, str]] = []
        for item in value:
            if isinstance(item, dict):
                pairs.append(
                    (int(item.get("language", -1)), str(item.get("solution", "")))
                )
        return pairs
    return []


def normalized_description_hash(value: str) -> str:
    normalized = " ".join(value.casefold().split())
    return sha256_text(normalized)


def humaneval_prompt_hashes(path: Path) -> set[str]:
    hashes: set[str] = set()
    if not path.is_file():
        return hashes
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                hashes.add(
                    normalized_description_hash(str(json.loads(line).get("prompt", "")))
                )
    return hashes


def collect_candidates(
    shards: list[Path],
    max_tests: int,
    max_solutions: int,
    max_description_chars: int,
    max_sequence_tokens: int,
) -> tuple[list[Candidate], Counter[str]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise CodeContestsError("pyarrow is required to read CodeContests") from exc
    rejected: Counter[str] = Counter()
    formal_hashes = humaneval_prompt_hashes(DEFAULT_HUMANEVAL)
    candidates: list[Candidate] = []
    columns = (
        "name",
        "description",
        "public_tests",
        "private_tests",
        "generated_tests",
        "source",
        "difficulty",
        "solutions",
        "input_file",
        "output_file",
    )
    for shard in shards:
        parquet = pq.ParquetFile(shard)
        for batch in parquet.iter_batches(batch_size=8, columns=list(columns)):
            for row in batch.to_pylist():
                description = str(row.get("description", "")).strip()
                if not description or len(description) > max_description_chars:
                    rejected["description_missing_or_too_long"] += 1
                    continue
                if normalized_description_hash(description) in formal_hashes:
                    rejected["humaneval_exact_prompt_overlap"] += 1
                    continue
                if str(row.get("input_file") or "").strip() or str(
                    row.get("output_file") or ""
                ).strip():
                    rejected["file_io_problem"] += 1
                    continue
                tests: list[dict[str, str]] = []
                for key in ("public_tests", "private_tests", "generated_tests"):
                    tests.extend(normalize_tests(row.get(key), limit=max_tests))
                    if len(tests) >= max_tests:
                        break
                tests = tests[:max_tests]
                if len(tests) < 3:
                    rejected["insufficient_tests"] += 1
                    continue
                python3: list[str] = []
                for language, source in solution_pairs(row.get("solutions")):
                    if language != 3 or not source.strip():
                        continue
                    safe, _ = safe_contest_source(source)
                    if safe:
                        python3.append(source.strip())
                    if len(python3) >= max_solutions:
                        break
                if not python3:
                    rejected["no_safe_python3_solution"] += 1
                    continue
                identity = stable_json(
                    {
                        "name": row.get("name"),
                        "description": description,
                        "source": row.get("source"),
                    }
                )
                group_id = f"code_contests/train/{sha256_text(identity)[:20]}"
                cache_key = sha256_text(
                    stable_json(
                        {
                            "group": group_id,
                            "tests": stable_test_hash(tests),
                            "solutions": [sha256_text(source) for source in python3],
                            "max_sequence_tokens": max_sequence_tokens,
                            "token_filter_protocol": "qwen3_chat_template_thinking_off_v1",
                        }
                    )
                )
                candidates.append(
                    Candidate(
                        cache_key=cache_key,
                        group_id=group_id,
                        name=str(row.get("name", "")),
                        description=description,
                        origin=str(row.get("source", "")),
                        difficulty=str(row.get("difficulty", "")),
                        tests=tuple(tests),
                        solutions=tuple(python3),
                    )
                )
    deduplicated = {
        candidate.group_id: candidate
        for candidate in sorted(candidates, key=lambda item: item.cache_key)
    }
    rejected["duplicate_problem_identity"] += len(candidates) - len(deduplicated)
    return list(deduplicated.values()), rejected


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return cache
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            cache[str(value.get("cache_key", ""))] = value
    return cache


def verified_row(
    candidate: Candidate,
    timeout_sec: float,
    tokenizer: Any,
    max_sequence_tokens: int,
) -> tuple[dict[str, Any] | None, str]:
    selected = ""
    last_reason = "no_solution_passed"
    passed_tests = 0
    prompt = (
        "Solve the programming problem below. Return only a complete Python 3 "
        "program that reads standard input and writes standard output. Do not use "
        "markdown or explanations.\n\n"
        f"{candidate.description}"
    )
    for solution in candidate.solutions:
        result = run_contest_tests(solution, candidate.tests, timeout_sec)
        if not result.passed:
            last_reason = result.reason
            continue
        tokenized = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": solution},
            ],
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        if len(tokenized) > max_sequence_tokens:
            last_reason = "sequence_too_long"
            continue
        selected = solution
        passed_tests = result.passed_tests
        break
    if not selected:
        return None, last_reason
    row = {
        "sample_id": f"p0a4r3/{candidate.group_id}",
        "validation_group_id": candidate.group_id,
        "dataset_key": "humaneval",
        "source": "p0a4r3_code_contests_train_verified_python3",
        "origin": "code_contests",
        "origin_source": candidate.origin,
        "origin_difficulty": candidate.difficulty,
        "messages": [{"role": "user", "content": prompt}],
        "answer": selected,
        "code_eval": {
            "kind": "code_contests_stdio_v1",
            "execution_protocol": "isolated_python3_stdio_resource_limited",
            "tests": list(candidate.tests),
            "tests_hash": stable_test_hash(candidate.tests),
            "verified_test_count": passed_tests,
        },
        "supervision_type": "canonical_executable_solution",
        "used_for_final_test": False,
    }
    return row, ""


def validate_all(
    candidates: list[Candidate],
    cache_path: Path,
    workers: int,
    timeout_sec: float,
    tokenizer: Any,
    max_sequence_tokens: int,
) -> tuple[list[dict[str, Any]], Counter[str], int]:
    cache = load_cache(cache_path)
    accepted: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    pending: list[Candidate] = []
    for candidate in candidates:
        cached = cache.get(candidate.cache_key)
        if cached is None:
            pending.append(candidate)
        elif isinstance(cached.get("row"), dict):
            accepted.append(cached["row"])
        else:
            rejected[str(cached.get("reason", "cached_rejection"))] += 1
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as stream:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    verified_row,
                    candidate,
                    timeout_sec,
                    tokenizer,
                    max_sequence_tokens,
                ): candidate
                for candidate in pending
            }
            for index, future in enumerate(as_completed(futures), start=1):
                candidate = futures[future]
                try:
                    row, reason = future.result()
                except Exception as exc:
                    row, reason = None, f"worker_error:{type(exc).__name__}"
                stream.write(
                    stable_json(
                        {
                            "cache_key": candidate.cache_key,
                            "group_id": candidate.group_id,
                            "row": row,
                            "reason": reason,
                        }
                    )
                    + "\n"
                )
                stream.flush()
                if row is not None:
                    accepted.append(row)
                else:
                    rejected[reason] += 1
                if index % 100 == 0 or index == len(pending):
                    print(
                        f"[CodeContests verify] {index}/{len(pending)} "
                        f"accepted={len(accepted)} rejected={sum(rejected.values())}",
                        flush=True,
                    )
    return accepted, rejected, len(candidates) - len(pending)


def build(args: argparse.Namespace) -> None:
    parquet_dir = resolve_path(args.parquet_dir)
    shards = sorted(parquet_dir.glob("train-*.parquet"))
    if not shards:
        raise CodeContestsError(
            f"No train parquet shards under {display_path(parquet_dir)}"
        )
    candidates, source_rejections = collect_candidates(
        shards,
        args.max_tests,
        args.max_solutions,
        args.max_description_chars,
        args.max_sequence_tokens,
    )
    ordered = sorted(
        candidates,
        key=lambda item: sha256_text(f"{args.seed}:{item.group_id}"),
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        resolve_path(args.tokenizer_dir),
        local_files_only=True,
        trust_remote_code=True,
    )
    required = args.train_target + args.validation_target
    verified: list[dict[str, Any]] = []
    execution_rejections: Counter[str] = Counter()
    cache_hits = 0
    examined = 0
    first_batch = required + 256
    while len(verified) < required and examined < len(ordered):
        batch_size = first_batch if examined == 0 else 256
        batch = ordered[examined : examined + batch_size]
        batch_verified, batch_rejections, batch_cache_hits = validate_all(
            batch,
            resolve_path(args.cache),
            args.workers,
            args.timeout_sec,
            tokenizer,
            args.max_sequence_tokens,
        )
        verified.extend(batch_verified)
        execution_rejections.update(batch_rejections)
        cache_hits += batch_cache_hits
        examined += len(batch)
    verified.sort(
        key=lambda row: sha256_text(
            f"{args.seed}:{row.get('validation_group_id', '')}"
        )
    )
    selected = verified[:required]
    validation = selected[: args.validation_target]
    train = selected[args.validation_target :]
    for row in train:
        row["used_for_training"] = True
        row["used_for_validation"] = False
    for row in validation:
        row["used_for_training"] = False
        row["used_for_validation"] = True
    train.sort(key=lambda row: str(row["sample_id"]))
    validation.sort(key=lambda row: str(row["sample_id"]))
    train_path = resolve_path(args.train_output)
    validation_path = resolve_path(args.validation_output)
    write_jsonl(train_path, train)
    write_jsonl(validation_path, validation)
    errors: list[str] = []
    if len(train) < args.train_target:
        errors.append("insufficient_verified_train_groups")
    if len(validation) < args.validation_target:
        errors.append("insufficient_verified_validation_groups")
    train_groups = {str(row["validation_group_id"]) for row in train}
    validation_groups = {str(row["validation_group_id"]) for row in validation}
    if train_groups & validation_groups:
        errors.append("train_validation_group_overlap")
    audit = {
        "gate": "P0-A4R3-CODE-CONTESTS",
        "check_version": "1.0",
        "created_by": "model_compression/build_p0a4r3_code_data.py",
        "created_ts": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not errors else "failed",
        "policy": {
            "upstream_split": "train_only",
            "validation_or_test_split_read": False,
            "solution_languages": ["PYTHON3"],
            "teacher_output_validation": "all selected stdio tests pass",
            "sequence_limit": (
                f"prompt plus verified answer <= {args.max_sequence_tokens} tokens"
            ),
            "formal_humaneval_use": "prompt-only exact decontamination",
            "formal_test_labels_used": False,
        },
        "parquet_shards": {
            display_path(path): sha256_file(path) for path in shards
        },
        "candidate_count": len(candidates),
        "execution_examined_count": examined,
        "source_rejections": dict(source_rejections),
        "execution_verified_count": len(verified),
        "execution_rejections": dict(execution_rejections),
        "cache_hit_count": cache_hits,
        "train_output": display_path(train_path),
        "train_hash": sha256_file(train_path),
        "train_unique_groups": len(train_groups),
        "validation_output": display_path(validation_path),
        "validation_hash": sha256_file(validation_path),
        "validation_unique_groups": len(validation_groups),
        "train_validation_overlap": len(train_groups & validation_groups),
        "errors": errors,
    }
    audit["report_hash"] = sha256_text(stable_json(audit))
    write_json(resolve_path(args.audit), audit)
    print(
        f"CodeContests train={len(train)} validation={len(validation)} "
        f"status={audit['status']}",
        flush=True,
    )
    if errors:
        raise CodeContestsError(str(errors))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and execution-verify train-only CodeContests Python3 data."
    )
    parser.add_argument("command", choices=("download", "build", "all", "status"))
    parser.add_argument("--parquet-dir", default=str(DEFAULT_PARQUET_DIR))
    parser.add_argument("--train-output", default=str(DEFAULT_TRAIN))
    parser.add_argument("--validation-output", default=str(DEFAULT_VALIDATION))
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--train-target", type=int, default=1000)
    parser.add_argument("--validation-target", type=int, default=256)
    parser.add_argument("--max-tests", type=int, default=8)
    parser.add_argument("--max-solutions", type=int, default=3)
    parser.add_argument("--max-description-chars", type=int, default=12000)
    parser.add_argument("--tokenizer-dir", default=str(DEFAULT_TOKENIZER))
    parser.add_argument("--max-sequence-tokens", type=int, default=1536)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--timeout-sec", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=20260727)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if (
            args.train_target <= 0
            or args.validation_target <= 0
            or args.max_tests < 3
            or args.max_solutions <= 0
            or args.workers <= 0
            or args.timeout_sec <= 0
            or args.max_sequence_tokens <= 0
        ):
            raise CodeContestsError("Invalid count/worker/timeout arguments")
        parquet_dir = resolve_path(args.parquet_dir)
        if args.command == "status":
            shards = sorted(parquet_dir.glob("train-*.parquet"))
            print(f"parquet_shards={len(shards)}")
            for path in shards:
                print(f"{display_path(path)} bytes={path.stat().st_size}")
            for value in (args.train_output, args.validation_output, args.audit):
                path = resolve_path(value)
                print(f"{display_path(path)} exists={path.is_file()}")
            return 0
        if args.command in {"download", "all"}:
            shards = download_train_parquet(parquet_dir)
            print(f"CodeContests train shards ready: {len(shards)}")
        if args.command in {"build", "all"}:
            build(args)
        return 0
    except (
        CodeContestsError,
        OSError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
    ) as exc:
        print(f"P0-A4R3 CodeContests failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
