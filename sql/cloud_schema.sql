CREATE DATABASE IF NOT EXISTS db4ai_edgeserve;
SET DATABASE = db4ai_edgeserve;

CREATE TABLE IF NOT EXISTS semantic_distill_trace (
    trace_id STRING PRIMARY KEY,
    schema_version STRING NOT NULL,
    created_by STRING NOT NULL,
    split STRING NOT NULL,
    source_dataset STRING NOT NULL,
    sample_hash STRING NOT NULL,
    created_ts TIMESTAMP NOT NULL DEFAULT current_timestamp(),
    task_id STRING NOT NULL,
    evidence_items_json STRING NOT NULL,
    decision_tuple_json STRING NOT NULL,
    teacher_trace_json STRING NOT NULL,
    student_probe_trace_json STRING NOT NULL,
    repair_trace_json STRING NOT NULL,
    quant_behavior_trace_json STRING NOT NULL
);

CREATE TABLE IF NOT EXISTS evidence_planner_trace (
    planner_trace_id STRING PRIMARY KEY,
    schema_version STRING NOT NULL,
    created_by STRING NOT NULL,
    split STRING NOT NULL,
    source_dataset STRING NOT NULL,
    sample_hash STRING NOT NULL,
    created_ts TIMESTAMP NOT NULL DEFAULT current_timestamp(),
    task_id STRING NOT NULL,
    evidence_candidates_json STRING NOT NULL,
    evidence_plan_json STRING NOT NULL,
    planner_model_hash STRING NOT NULL,
    calibration_snapshot_hash STRING NOT NULL
);

CREATE TABLE IF NOT EXISTS runtime_state_trace (
    task_id STRING PRIMARY KEY,
    schema_version STRING NOT NULL,
    created_by STRING NOT NULL,
    split STRING NOT NULL,
    source_dataset STRING NOT NULL,
    sample_hash STRING NOT NULL,
    created_ts TIMESTAMP NOT NULL DEFAULT current_timestamp(),
    edge_node_id STRING NOT NULL,
    network_snapshot_json STRING NOT NULL,
    queue_state_json STRING NOT NULL,
    model_health_json STRING NOT NULL,
    context_state_json STRING NOT NULL,
    outbox_state_json STRING NOT NULL,
    task_risk STRING NOT NULL,
    runtime_latent_state_hash STRING NOT NULL,
    predicted_path_outcomes_json STRING NOT NULL
);

CREATE TABLE IF NOT EXISTS policy_action_trace (
    policy_trace_id STRING PRIMARY KEY,
    schema_version STRING NOT NULL,
    created_by STRING NOT NULL,
    split STRING NOT NULL,
    source_dataset STRING NOT NULL,
    sample_hash STRING NOT NULL,
    created_ts TIMESTAMP NOT NULL DEFAULT current_timestamp(),
    task_id STRING NOT NULL,
    runtime_state_hash STRING NOT NULL,
    selected_path STRING NOT NULL,
    predicted_outcome_distribution_json STRING NOT NULL,
    actual_outcome_json STRING NOT NULL,
    policy_snapshot_hash STRING NOT NULL
);

CREATE TABLE IF NOT EXISTS decision_tuple_trace (
    decision_tuple_id STRING PRIMARY KEY,
    schema_version STRING NOT NULL,
    created_by STRING NOT NULL,
    split STRING NOT NULL,
    source_dataset STRING NOT NULL,
    sample_hash STRING NOT NULL,
    created_ts TIMESTAMP NOT NULL DEFAULT current_timestamp(),
    task_id STRING NOT NULL,
    scene STRING NOT NULL,
    dataset STRING NOT NULL,
    edge_node_id STRING NOT NULL,
    decision_ts TIMESTAMP NOT NULL,
    object_id STRING NOT NULL,
    region_id STRING NOT NULL,
    event_id STRING NOT NULL,
    relation_group_id STRING NOT NULL,
    conflict_group_id STRING NOT NULL,
    network_profile STRING NOT NULL,
    workload_profile STRING NOT NULL,
    object_state_json STRING NOT NULL,
    event_type STRING NOT NULL,
    risk_attr STRING NOT NULL,
    action STRING NOT NULL,
    confidence FLOAT8 NOT NULL,
    review_intent STRING NOT NULL,
    is_provisional BOOL NOT NULL DEFAULT false,
    selected_path STRING NOT NULL,
    source STRING NOT NULL
);

CREATE TABLE IF NOT EXISTS relation_graph_trace (
    graph_id STRING PRIMARY KEY,
    schema_version STRING NOT NULL,
    created_by STRING NOT NULL,
    split STRING NOT NULL,
    source_dataset STRING NOT NULL,
    sample_hash STRING NOT NULL,
    created_ts TIMESTAMP NOT NULL DEFAULT current_timestamp(),
    nodes_json STRING NOT NULL,
    edges_json STRING NOT NULL,
    graph_model_hash STRING NOT NULL,
    graph_input_hash STRING NOT NULL,
    source_split STRING NOT NULL
);

CREATE TABLE IF NOT EXISTS conflict_inference_trace (
    conflict_inference_id STRING PRIMARY KEY,
    schema_version STRING NOT NULL,
    created_by STRING NOT NULL,
    split STRING NOT NULL,
    source_dataset STRING NOT NULL,
    sample_hash STRING NOT NULL,
    created_ts TIMESTAMP NOT NULL DEFAULT current_timestamp(),
    relation_group_id STRING NOT NULL,
    conflict_group_id STRING NOT NULL,
    conflict_type_distribution_json STRING NOT NULL,
    global_decision_distribution_json STRING NOT NULL,
    final_global_decision_json STRING NOT NULL,
    conflict_gt STRING NOT NULL,
    label_source STRING NOT NULL,
    conflict_gt_manifest_hash STRING NOT NULL,
    graph_model_hash STRING NOT NULL,
    event_id STRING NOT NULL
);

CREATE TABLE IF NOT EXISTS trust_posterior_trace (
    edge_node_id STRING PRIMARY KEY,
    schema_version STRING NOT NULL,
    created_by STRING NOT NULL,
    split STRING NOT NULL,
    source_dataset STRING NOT NULL,
    sample_hash STRING NOT NULL,
    created_ts TIMESTAMP NOT NULL DEFAULT current_timestamp(),
    task_type STRING NOT NULL,
    posterior_state_json STRING NOT NULL,
    posterior_snapshot_hash STRING NOT NULL,
    correction_history_hash STRING NOT NULL,
    ack_state STRING NOT NULL,
    recurrence_state STRING NOT NULL,
    network_profile STRING NOT NULL,
    workload_profile STRING NOT NULL
);

CREATE TABLE IF NOT EXISTS edge_outbox (
    outbox_id STRING PRIMARY KEY,
    schema_version STRING NOT NULL,
    created_by STRING NOT NULL,
    created_ts TIMESTAMP NOT NULL DEFAULT current_timestamp(),
    source_edge_node_id STRING NOT NULL,
    target STRING NOT NULL,
    payload_type STRING NOT NULL,
    payload_hash STRING NOT NULL,
    payload_json STRING NOT NULL,
    status STRING NOT NULL,
    retry_count INT8 NOT NULL DEFAULT 0,
    last_error STRING NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_runtime_state_edge_node ON runtime_state_trace (edge_node_id);
CREATE INDEX IF NOT EXISTS idx_decision_tuple_conflict_group ON decision_tuple_trace (conflict_group_id);
CREATE INDEX IF NOT EXISTS idx_conflict_inference_group ON conflict_inference_trace (conflict_group_id);
CREATE INDEX IF NOT EXISTS idx_edge_outbox_status ON edge_outbox (status);
