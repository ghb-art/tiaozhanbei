from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA = ROOT / "sql" / "cloud_schema.sql"
DEFAULT_SCHEMA_REPORT = ROOT / "reports" / "audit" / "gate_db_schema_check.json"
DEFAULT_CSV_OUTPUT = ROOT / "reports" / "audit" / "gate_db_smoke.csv"

REQUIRED_TABLES = [
    "semantic_distill_trace",
    "evidence_planner_trace",
    "runtime_state_trace",
    "policy_action_trace",
    "decision_tuple_trace",
    "relation_graph_trace",
    "conflict_inference_trace",
    "trust_posterior_trace",
    "edge_outbox",
]

PLACEHOLDER_MARKERS = ["TODO", "TBD", "placeholder", "...", "实际值"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def schema_errors(schema_path: Path) -> list[str]:
    if not schema_path.is_file():
        return [f"Missing schema file: {schema_path}"]

    text = schema_path.read_text(encoding="utf-8")
    lowered = text.lower()
    errors: list[str] = []

    if "create database if not exists db4ai_edgeserve" not in lowered:
        errors.append("Schema must create database db4ai_edgeserve")

    for table in REQUIRED_TABLES:
        if f"create table if not exists {table}" not in lowered:
            errors.append(f"Schema missing required table: {table}")

    for marker in PLACEHOLDER_MARKERS:
        if marker.lower() in lowered:
            errors.append(f"Schema contains placeholder marker: {marker}")

    return errors


def run_offline_schema_check(schema_path: Path, report_path: Path) -> int:
    errors = schema_errors(schema_path)
    report = {
        "check_version": "1.0",
        "created_by": "scripts/verify_gate_db.py",
        "created_ts": datetime.now(timezone.utc).isoformat(),
        "mode": "offline_schema_check",
        "schema_path": display_path(schema_path),
        "schema_hash": sha256_file(schema_path) if schema_path.is_file() else "",
        "required_tables": REQUIRED_TABLES,
        "live_gate_pass": False,
        "status": "failed" if errors else "passed",
        "errors": errors,
    }
    write_json(report_path, report)

    if errors:
        print("G-DB schema check failed:")
        for error in errors:
            print(f"[FAIL] {error}")
        print(f"Wrote {display_path(report_path)}")
        return 1

    print("G-DB schema check passed.")
    print(f"Wrote {display_path(report_path)}")
    return 0


def run_command(command: list[str], sql_input: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        input=sql_input,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def require_docker(container: str) -> list[str]:
    errors: list[str] = []
    if shutil.which("docker") is None:
        return ["Docker CLI is not available on PATH"]

    result = run_command(["docker", "inspect", "-f", "{{.State.Running}}", container])
    if result.returncode != 0:
        errors.append(f"Docker container is not available: {container}")
    elif result.stdout.strip().lower() != "true":
        errors.append(f"Docker container is not running: {container}")
    return errors


def run_kwdb_sql(container: str, kwbase_bin: str, sql: str, csv_format: bool = False) -> subprocess.CompletedProcess[str]:
    command = [
        "docker",
        "exec",
        "-i",
        container,
        kwbase_bin,
        "sql",
        "--insecure",
        "--host=127.0.0.1",
    ]
    if csv_format:
        command.append("--format=csv")
    return run_command(command, sql)


def smoke_sql(database: str) -> str:
    return f"""
SET DATABASE = {database};

DELETE FROM runtime_state_trace WHERE task_id = 'gdb_smoke_task';
DELETE FROM edge_outbox WHERE outbox_id = 'gdb_smoke_outbox';

INSERT INTO runtime_state_trace (
    task_id, schema_version, created_by, split, source_dataset, sample_hash,
    edge_node_id, network_snapshot_json, queue_state_json, model_health_json,
    context_state_json, outbox_state_json, task_risk, runtime_latent_state_hash,
    predicted_path_outcomes_json
) VALUES (
    'gdb_smoke_task', '1.0', 'scripts/verify_gate_db.py', 'validation', 'G-DB smoke',
    'smoke_sample_hash', 'edge_smoke_001', '{{"rtt_ms": 10}}', '{{"pending_task_count": 0}}',
    '{{"server_healthy": true}}', '{{"context_complete": true}}', '{{"backlog": 0}}',
    'low', 'runtime_latent_state_smoke_hash', '{{"edge_only": {{"p95_ms": 50}}}}'
);

INSERT INTO edge_outbox (
    outbox_id, schema_version, created_by, source_edge_node_id, target,
    payload_type, payload_hash, payload_json, status
) VALUES (
    'gdb_smoke_outbox', '1.0', 'scripts/verify_gate_db.py', 'edge_smoke_001',
    'kwdb-cloud', 'runtime_state_trace', 'smoke_payload_hash',
    '{{"task_id": "gdb_smoke_task"}}', 'pending'
);
"""


def smoke_select_sql(database: str) -> str:
    return f"""
SET DATABASE = {database};
SELECT task_id, edge_node_id, split, source_dataset FROM runtime_state_trace WHERE task_id = 'gdb_smoke_task';
SELECT outbox_id, source_edge_node_id, target, status FROM edge_outbox WHERE outbox_id = 'gdb_smoke_outbox';
"""


def run_live_gate(schema_path: Path, container: str, kwbase_bin: str, database: str, csv_output: Path) -> int:
    errors = schema_errors(schema_path)
    errors.extend(require_docker(container))
    if errors:
        print("G-DB live gate precheck failed:")
        for error in errors:
            print(f"[FAIL] {error}")
        return 1

    schema_result = run_kwdb_sql(container, kwbase_bin, schema_path.read_text(encoding="utf-8"))
    if schema_result.returncode != 0:
        print("Failed to apply cloud schema:")
        print(schema_result.stderr or schema_result.stdout)
        return schema_result.returncode

    smoke_result = run_kwdb_sql(container, kwbase_bin, smoke_sql(database))
    if smoke_result.returncode != 0:
        print("Failed to run DB smoke write/query setup:")
        print(smoke_result.stderr or smoke_result.stdout)
        return smoke_result.returncode

    csv_result = run_kwdb_sql(container, kwbase_bin, smoke_select_sql(database), csv_format=True)
    if csv_result.returncode != 0:
        print("Failed to export DB smoke query CSV:")
        print(csv_result.stderr or csv_result.stdout)
        return csv_result.returncode

    csv_output.parent.mkdir(parents=True, exist_ok=True)
    csv_lines = [line for line in csv_result.stdout.splitlines() if line.strip() != "SET"]
    csv_output.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    print("G-DB live gate passed.")
    print(f"Wrote {display_path(csv_output)}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify G-DB schema and KWDB smoke gate.")
    parser.add_argument("--schema", default=str(DEFAULT_SCHEMA), help="Path to cloud_schema.sql.")
    parser.add_argument("--container", default="kwdb-cloud", help="KWDB Docker container name.")
    parser.add_argument("--kwbase-bin", default="/kaiwudb/bin/kwbase", help="Path to kwbase binary inside the container.")
    parser.add_argument("--database", default="db4ai_edgeserve", help="Database name used by the schema.")
    parser.add_argument("--csv-output", default=str(DEFAULT_CSV_OUTPUT), help="CSV output path for live smoke query.")
    parser.add_argument(
        "--offline-schema-check",
        action="store_true",
        help="Only validate the schema file and write an offline report; does not require Docker.",
    )
    parser.add_argument("--schema-report", default=str(DEFAULT_SCHEMA_REPORT), help="Offline schema-check report path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    schema_path = Path(args.schema)
    if not schema_path.is_absolute():
        schema_path = ROOT / schema_path

    if args.offline_schema_check:
        report_path = Path(args.schema_report)
        if not report_path.is_absolute():
            report_path = ROOT / report_path
        return run_offline_schema_check(schema_path, report_path)

    csv_output = Path(args.csv_output)
    if not csv_output.is_absolute():
        csv_output = ROOT / csv_output
    return run_live_gate(schema_path, args.container, args.kwbase_bin, args.database, csv_output)


if __name__ == "__main__":
    sys.exit(main())
