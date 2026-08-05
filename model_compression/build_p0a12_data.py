#!/usr/bin/env python3
"""Build the leak-safe P0-A12 MetaMath train set and SVAMP holdout."""

from __future__ import annotations

import argparse
import codecs
import hashlib
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterator
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p0a12_math_repair.json"
TOKENIZER_DIR = ROOT / "models/checkpoints/p0a4/student-shared-merged"
P0A11_TRAIN = ROOT / "data/p0a11/math_train.jsonl"
OUTPUT_DIR = ROOT / "data/p0a12"
AUDIT = ROOT / "reports/audit/gate_p0a12_data.json"
METAMATH_REVISION = "aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18"
METAMATH_URL = (
    "https://huggingface.co/datasets/meta-math/MetaMathQA/resolve/"
    f"{METAMATH_REVISION}/MetaMathQA-395K.json"
)
SVAMP_REVISION = "689d7ccac74b9983a2ac7cc3b264f441b99e7c53"
SVAMP_URL = f"https://raw.githubusercontent.com/arkilpatel/SVAMP/{SVAMP_REVISION}/SVAMP.json"
SVAMP_LICENSE_URL = f"https://raw.githubusercontent.com/arkilpatel/SVAMP/{SVAMP_REVISION}/LICENSE"
ACCEPTED_TYPES = ("GSM_Rephrased", "GSM_AnsAug", "GSM_SV")
METAMATH_TARGET = 16000
MAX_SEQUENCE_LENGTH = 1536
SEED = 20260803
HISTORY = (
    ROOT / "data/p0a6/quick_validation.jsonl",
    ROOT / "data/p0a6/full_validation.jsonl",
    ROOT / "data/p0a10/math_validation.jsonl",
    ROOT / "data/p0a11/math_validation.jsonl",
)
SYSTEM_PROMPT = (
    "Solve the problem concisely. End with one line formatted as `#### 42`, "
    "where 42 is replaced by the actual numeric answer."
)
NUMBER = re.compile(r"####\s*([-+]?(?:\d[\d,]*(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)")


class BuildError(RuntimeError):
    pass


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(value: str, namespace: str) -> str:
    return sha256_text(f"{SEED}:{namespace}:{value}")


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def normalize_number(value: Any) -> str:
    cleaned = str(value).replace(",", "").strip()
    try:
        number = Decimal(cleaned)
    except InvalidOperation as exc:
        raise BuildError(f"Invalid numeric answer: {value!r}") from exc
    if not number.is_finite():
        raise BuildError(f"Non-finite answer: {value!r}")
    result = format(number.normalize(), "f")
    return "0" if Decimal(result) == 0 else result


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise BuildError(f"Missing input: {path.relative_to(ROOT)}")
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(value)
    temporary.replace(path)


def download(url: str, maximum_bytes: int) -> bytes:
    request = Request(url, headers={"User-Agent": "p0a12-data-builder/1.0"})
    with urlopen(request, timeout=120) as response:
        value = response.read(maximum_bytes + 1)
    if len(value) > maximum_bytes:
        raise BuildError(f"Download exceeds safety cap: {url}")
    return value


def stream_json_array(url: str) -> Iterator[dict[str, Any]]:
    """Incrementally parse one remote JSON array without storing the 396 MB source."""
    request = Request(url, headers={"User-Agent": "p0a12-data-builder/1.0"})
    decoder = json.JSONDecoder()
    utf8 = codecs.getincrementaldecoder("utf-8")()
    buffer = ""
    position = 0
    started = False
    finished = False

    def consume(final: bool = False) -> Iterator[dict[str, Any]]:
        nonlocal buffer, position, started, finished
        while True:
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if not started:
                if position >= len(buffer):
                    break
                if buffer[position] != "[":
                    raise BuildError("MetaMath source is not a JSON array")
                position += 1
                started = True
                continue
            while position < len(buffer) and (buffer[position].isspace() or buffer[position] == ","):
                position += 1
            if position < len(buffer) and buffer[position] == "]":
                position += 1
                finished = True
                break
            if position >= len(buffer):
                break
            try:
                item, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                if final:
                    raise BuildError("Truncated MetaMath JSON source")
                break
            if not isinstance(item, dict):
                raise BuildError("MetaMath array contains a non-object")
            position = end
            yield item
        if position > 1024 * 1024:
            buffer = buffer[position:]
            position = 0

    with urlopen(request, timeout=120) as response:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            buffer += utf8.decode(chunk)
            yield from consume(False)
        buffer += utf8.decode(b"", final=True)
        yield from consume(True)
    if not started or not finished:
        raise BuildError("MetaMath JSON array did not terminate cleanly")


def historical_prompts() -> set[str]:
    prompts: set[str] = set()
    for path in HISTORY:
        for row in read_jsonl(path):
            if str(row.get("domain", "")) == "math" or row.get("dataset_key") == "gsm8k":
                prompt = str(row.get("prompt", ""))
                if prompt:
                    prompts.add(normalize_text(prompt))
    return prompts


def normalize_solution(response: str) -> tuple[str, str] | None:
    matches = list(NUMBER.finditer(response))
    if not matches:
        return None
    match = matches[-1]
    try:
        reference = normalize_number(match.group(1))
    except BuildError:
        return None
    reasoning = response[:match.start()].strip()
    if not reasoning or len(reasoning) > 5500:
        return None
    return f"{reasoning}\n#### {reference}", reference


def token_length(tokenizer: Any, row: dict[str, Any]) -> int:
    return len(tokenizer.apply_chat_template(
        row["messages"] + [{"role": "assistant", "content": row["answer"]}],
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False,
    ))


def metamath_row(raw: dict[str, Any]) -> dict[str, Any] | None:
    row_type = str(raw.get("type", ""))
    if row_type not in ACCEPTED_TYPES:
        return None
    query = str(raw.get("query", "")).strip()
    original = str(raw.get("original_question", "")).strip()
    response = str(raw.get("response", ""))
    if not query or not original or len(query) > 2600:
        return None
    solution = normalize_solution(response)
    if solution is None:
        return None
    answer, reference = solution
    query_hash = sha256_text(normalize_text(query))
    original_hash = sha256_text(normalize_text(original))
    return {
        "sample_id": f"metamathqa/{row_type}/{query_hash[:24]}",
        "dataset_key": "metamathqa",
        "domain": "math",
        "task_id": "math",
        "source": f"meta-math/MetaMathQA@{METAMATH_REVISION}",
        "split_role": "train",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        "answer": answer,
        "metadata": {
            "reference_answer": reference,
            "metamath_type": row_type,
            "original_question_hash": original_hash,
            "query_hash": query_hash,
        },
        "answer_token_weight": 1.0,
        "quality_weight": 1.0,
        "training_weight": 1.0,
        "kl_weight": 0.15,
    }


def build_metamath(tokenizer: Any, history: set[str], replay_prompts: set[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    buckets: dict[str, dict[str, tuple[str, dict[str, Any]]]] = {}
    counts: Counter[str] = Counter()
    streamed = 0
    for raw in stream_json_array(METAMATH_URL):
        streamed += 1
        if streamed % 50000 == 0:
            print(f"[MetaMath stream] {streamed}", flush=True)
        row = metamath_row(raw)
        if row is None:
            counts["shape_or_answer"] += 1
            continue
        query_norm = normalize_text(str(row["messages"][1]["content"]))
        original_norm = normalize_text(str(raw.get("original_question", "")))
        if query_norm in history or original_norm in history:
            counts["historical_validation"] += 1
            continue
        if query_norm in replay_prompts:
            counts["exact_replay_duplicate"] += 1
            continue
        metadata = row["metadata"]
        original_hash = str(metadata["original_question_hash"])
        row_type = str(metadata["metamath_type"])
        key = stable_key(str(metadata["query_hash"]), "metamath-candidate")
        per_original = buckets.setdefault(original_hash, {})
        previous = per_original.get(row_type)
        if previous is None or key < previous[0]:
            per_original[row_type] = (key, row)
    candidates = [entry[1] for values in buckets.values() for entry in values.values()]
    candidates.sort(key=lambda row: stable_key(str(row["sample_id"]), "metamath-select"))
    selected: list[dict[str, Any]] = []
    token_rejected = 0
    for row in candidates:
        if token_length(tokenizer, row) > MAX_SEQUENCE_LENGTH:
            token_rejected += 1
            continue
        selected.append(row)
        if len(selected) == METAMATH_TARGET:
            break
    if len(selected) != METAMATH_TARGET:
        raise BuildError(f"Only {len(selected)} MetaMath rows passed; need {METAMATH_TARGET}")
    return selected, {
        "streamed_rows": streamed,
        "unique_originals": len(buckets),
        "candidates_before_token_scan": len(candidates),
        "token_rejected_before_target": token_rejected,
        "selected": len(selected),
        "rejections": dict(sorted(counts.items())),
    }


def build_svamp() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw = download(SVAMP_URL, 5 * 1024 * 1024)
    license_bytes = download(SVAMP_LICENSE_URL, 1024 * 1024)
    try:
        source = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BuildError(f"Invalid SVAMP JSON: {exc}") from exc
    if not isinstance(source, list) or len(source) != 1000:
        raise BuildError(f"Unexpected SVAMP row count: {len(source) if isinstance(source, list) else 'not-list'}")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in source:
        sample_id = f"svamp/{str(item.get('ID', '')).strip()}"
        if sample_id in seen or sample_id == "svamp/":
            raise BuildError(f"Duplicate/missing SVAMP id: {sample_id}")
        seen.add(sample_id)
        body = str(item.get("Body", "")).strip()
        question = str(item.get("Question", "")).strip()
        if not body or not question:
            raise BuildError(f"SVAMP row lacks prompt: {sample_id}")
        rows.append({
            "sample_id": sample_id,
            "dataset_key": "svamp",
            "domain": "math",
            "source": f"arkilpatel/SVAMP@{SVAMP_REVISION}",
            "split_role": "p0a12_external_validation",
            "prompt": f"{body} {question}",
            "reference": normalize_number(item.get("Answer")),
            "validator": "exact_numeric_answer",
            "unit_tests": [],
            "metadata": {"equation": str(item.get("Equation", "")), "type": str(item.get("Type", ""))},
        })
    source_dir = OUTPUT_DIR / "sources"
    write_bytes(source_dir / "SVAMP.json", raw)
    write_bytes(source_dir / "SVAMP.LICENSE", license_bytes)
    return rows, {
        "source_sha256": sha256_bytes(raw),
        "license_sha256": sha256_bytes(license_bytes),
        "revision": SVAMP_REVISION,
    }


def build() -> int:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise BuildError("transformers is required") from exc
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config.get("protocol") != "P0-A12-METAMATH-REPAIR":
        raise BuildError("P0-A12 config identity mismatch")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True, trust_remote_code=True)
    p0a11 = read_jsonl(P0A11_TRAIN)
    errors = [row for row in p0a11 if row.get("hard_mining_role") == "base_error"]
    replay = [row for row in p0a11 if row.get("hard_mining_role") == "correct_replay"]
    if len(errors) != 1424 or len(replay) != 712:
        raise BuildError(f"Unexpected P0-A11 replay counts: errors={len(errors)} replay={len(replay)}")
    for row in errors + replay:
        row["kl_weight"] = 0.15
        row["training_weight"] = 2.0 if row.get("hard_mining_role") == "base_error" else 0.5
        row["p0a12_role"] = "hard_error" if row.get("hard_mining_role") == "base_error" else "correct_replay"
    replay_prompts = {normalize_text(str(row["messages"][1]["content"])) for row in errors + replay}
    history = historical_prompts()
    metamath, metamath_stats = build_metamath(tokenizer, history, replay_prompts)
    train = metamath + errors + replay
    train.sort(key=lambda row: stable_key(str(row["sample_id"]), "p0a12-train"))
    if len({str(row["sample_id"]) for row in train}) != len(train):
        raise BuildError("P0-A12 train has duplicate ids")
    maximum_tokens = max(token_length(tokenizer, row) for row in train)
    if maximum_tokens > MAX_SEQUENCE_LENGTH:
        raise BuildError(f"P0-A12 token budget exceeded: {maximum_tokens}")
    svamp, svamp_meta = build_svamp()
    train_prompts = {normalize_text(str(row["messages"][1]["content"])) for row in train}
    validation_prompts = {normalize_text(str(row["prompt"])) for row in svamp}
    if train_prompts & validation_prompts:
        raise BuildError("P0-A12 train-SVAMP exact prompt overlap")
    outputs = {
        "math_train": train,
        "math_validation": svamp,
    }
    output_meta: dict[str, Any] = {}
    for name, rows in outputs.items():
        path = OUTPUT_DIR / f"{name}.jsonl"
        write_jsonl(path, rows)
        output_meta[name] = {"path": path.relative_to(ROOT).as_posix(), "rows": len(rows), "sha256": sha256_file(path)}
    audit = {
        "gate": "P0-A12-METAMATH-DATA",
        "check_version": "1.0",
        "created_by": "model_compression/build_p0a12_data.py",
        "created_ts": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "policy": {
            "gate300_opened": False,
            "formal_full_opened": False,
            "formal_test_rows_used": 0,
            "p0a11_validation_reused": False,
            "svamp_used_for_training": False,
        },
        "counts": {
            "metamath_train": len(metamath),
            "hard_errors": len(errors),
            "correct_replay": len(replay),
            "math_train": len(train),
            "svamp_validation": len(svamp),
        },
        "metamath": {
            "repo": "meta-math/MetaMathQA",
            "revision": METAMATH_REVISION,
            "file": "MetaMathQA-395K.json",
            "raw_file_persisted": False,
            "accepted_types": list(ACCEPTED_TYPES),
            **metamath_stats,
        },
        "svamp": svamp_meta,
        "token_scan": {"status": "passed", "maximum_observed_tokens": maximum_tokens, "max_sequence_length": MAX_SEQUENCE_LENGTH},
        "history_hashes": {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in HISTORY},
        "inputs": {
            CONFIG.relative_to(ROOT).as_posix(): sha256_file(CONFIG),
            P0A11_TRAIN.relative_to(ROOT).as_posix(): sha256_file(P0A11_TRAIN),
        },
        "outputs": output_meta,
        "errors": [],
    }
    audit["report_hash"] = sha256_text(json.dumps(audit, ensure_ascii=False, sort_keys=True))
    write_json(AUDIT, audit)
    print(f"P0-A12 data passed train={len(train)} MetaMath={len(metamath)} SVAMP={len(svamp)}")
    print(f"Wrote {AUDIT.relative_to(ROOT)} status=passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "status"))
    args = parser.parse_args()
    if args.command == "build":
        return build()
    if not AUDIT.is_file():
        raise BuildError("P0-A12 audit is missing")
    value = json.loads(AUDIT.read_text(encoding="utf-8"))
    print(json.dumps({"status": value.get("status"), "counts": value.get("counts")}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BuildError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"P0-A12 data build failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
