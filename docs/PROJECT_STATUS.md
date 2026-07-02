# Project Status

更新时间：2026-07-02

当前阶段：G-KD-TRACE 已按第 2 章主实验口径完成全量 train split teacher trace gate，并完成 Student-Base probe / repair mining smoke；下一步训练 CEDD-Structured adapter、运行正式 student probe，并实现 GSM8K/HumanEval/CMMLU 能力保持率评测脚本。

## 当前结论

项目已经有完整的研究方案、目录规划、本地数据 payload、冻结 split、正式 manifest、conflict audit 文件、可运行的 KWDB/KaiwuDB 单节点验证、通过 live gate 的 14B-AWQ vLLM 云端教师服务验证、通过 gate 的第 2 章主实验全量 teacher trace，以及可运行的 student probe / repair mining smoke 链路。当前主实验数据集固定为 `GSM8K`、`HumanEval`、`CMMLU`、`NEU-DET`、`CityFlow`；`MMLU`、`MVTec AD`、`UA-DETRAC` 保留为支持/备份资产，不进入主实验。

第 2 章当前已闭合的是任务表征压缩的数据生成链路：teacher trace 只使用有 train split 且属于主实验的 `GSM8K`、`NEU-DET`、`CityFlow`。`HumanEval` 和 `CMMLU` 只用于后续能力保持率评测，不进入蒸馏训练。

## 已完成

- Git 仓库已初始化，并存在初始提交。
- 已建立挑战杯项目目录结构。
- 已有 `README.md` 和 `IMPLEMENTATION_PLAN.md`。
- 已加入基础 `.gitignore`，避免提交数据、模型、日志和运行产物。
- 已新增基础配置文件和结构校验脚本。
- 已按 `IMPLEMENTATION_PLAN.md` 生成三类 manifest 模板：`dataset_manifest.template.json`、`manifest.template.json`、`conflict_gt_manifest.template.json`。
- 已新增 `scripts/generate_manifest_template.py`，用于重新生成 manifest 模板。
- `scripts/validate_project_structure.py` 已检查 manifest 模板存在、可解析且标记为 `template_only=true`。
- 已新增 `scripts/validate_manifest_files.py`，用于验收正式 `dataset_manifest.json`、`manifest.json`、`conflict_gt_manifest.json`。
- 已新增 `scripts/inspect_datasets.py`，用于扫描 `data/datasets/` 并输出数据资产盘点报告。
- 已重新生成 `reports/preflight/data_inventory.json`。当前 8 个预期数据集目录均存在且均有 payload。
- 已新增 `docs/DATASET_SOURCES.md`，记录 8 个目标数据集的用途、放置目录、接入注意事项和待核验来源状态。
- 已核验并补充 `docs/DATASET_SOURCES.md` 中 8 个目标数据集的官方/主来源、访问方式、license/terms 注意事项和下载前人工确认要求。
- 已新增 `scripts/validate_dataset_presence.py`，用于检查数据集目录是否存在 `.gitkeep` 之外的真实 payload 文件。
- 已新增 `scripts/preflight_runtime_smoke.py`，用于检查 Python 版本、关键目录、配置/模板可读性，并调用现有轻量校验脚本完成 runtime smoke。
- 已生成 `reports/preflight/runtime_smoke.json`，作为当前工程骨架可运行性的预检报告。
- 已接入 GSM8K、HumanEval、MMLU、CMMLU 的本地数据文件；MMLU 官方 `data.tar` 已展开到 `data/datasets/mmlu/data/`。
- 已将 MVTec AD、CityFlow、UA-DETRAC 大型下载包以硬链接归档方式挂载到对应 `data/datasets/` 目录，避免重复占用磁盘空间。
- 已解压 NEU-DET Kaggle mirror 到 `data/datasets/neu_det/`，并记录其 split 中 `crazing_240` 图像/标注错位问题。
- 已展开 MVTec AD、CityFlow、UA-DETRAC 到标准子目录。
- 已将第 2 章主实验口径固定为 `GSM8K`、`HumanEval`、`CMMLU`、`NEU-DET`、`CityFlow`；`MMLU`、`MVTec AD`、`UA-DETRAC` 不进入主实验。
- 已新增 `docs/DATASET_SPLIT_STRATEGY.md`，记录 8 个数据集的固定划分方法。
- 已新增 `scripts/validate_splits.py`，生成并验证 `data/splits/frozen_splits.json` 和各数据集 split id 文件。
- 已新增 `scripts/generate_formal_manifests.py`，生成正式 `dataset_manifest.json`、`manifest.json`、`conflict_gt_manifest.json`。
- 已新增 `scripts/setup_datasets.sh`，用于检查本地 8 个 G-DATA 数据集 payload 是否放在标准目录。
- 已新增 `scripts/build_conflict_gt.py`，基于冻结 split 生成可追溯到 CityFlow、MVTec AD、NEU-DET test 样本的 conflict ground truth manifest。
- 已生成 `reports/audit/conflict_gt_audit.csv` 和 `reports/audit/conflict_gt_sample_audit.json`，用于完整审计与抽样审计 conflict ground truth。
- 已通过 `python scripts/validate_manifest_files.py --strict-gdata` 验收正式 manifest。
- 已新增 `sql/cloud_schema.sql`，覆盖 8 类核心 trace 表和 `edge_outbox` 表。
- 已新增 `docker/docker-compose.kwdb.yml`，用于启动 KWDB/KaiwuDB 单节点容器。
- 已新增 `scripts/verify_gate_db.py`，支持 G-DB live gate 和离线 schema 检查。
- 已通过 `python scripts/verify_gate_db.py --offline-schema-check` 完成 G-DB schema 离线检查，并生成 `reports/audit/gate_db_schema_check.json`。
- 已启动 `kwdb-cloud` Docker 容器，并通过 `python scripts/verify_gate_db.py` 完成 schema 应用、写入、查询和 CSV 导出 live gate。
- 已生成 `reports/audit/gate_db_smoke.csv`，记录 G-DB smoke 查询与 outbox 查询结果。
- 已新增 `scripts/download_models.py`，并将 G-CLOUD/G-KD 所需的三份 Qwen 模型下载到本地 `models/pretrained/`。
- 已生成 `reports/audit/model_downloads.json`，记录本地模型目录、文件数、体积和权重文件 SHA256，供上传服务器后核对。
- 已在项目 `.venv` 中安装并验证 `vllm==0.8.5`、`torch==2.6.0+cu124`、`transformers==4.51.3`。
- 已新增 `scripts/verify_gate_cloud.py`，支持本地模型审计、`/health` 检查、OpenAI-compatible streaming smoke 和首 token 延迟统计。
- 已通过 14B-AWQ G-CLOUD live gate：`/health` 200，smoke 首 token 延迟 0.581s，小于 2s 阈值。
- 已生成 `reports/audit/gate_cloud_smoke.json`，记录 `model_hash`、`prompt_hash`、健康检查、服务模型 id 和 smoke 响应。
- 已将 G-CLOUD 审计 hash、模型 hash、prompt hash 和首 token 延迟写入 `manifest.json`。
- 已新增 `model_compression/generate_teacher_traces.py`，从冻结 split 读取样本，调用 OpenAI-compatible vLLM 接口生成结构化 teacher decision trace 和 distill JSONL。
- 已通过 32 条 G-KD-TRACE teacher smoke，覆盖 5 个有 train split 的数据集：`teacher_health_status=200`，`successful_trace_count=32`，`failed_trace_count=0`，`parse_success_rate=1.0`。
- 已生成 `reports/audit/gate_kd_trace_teacher_smoke.json`，并将 smoke 审计 hash、trace hash、distill hash、sample ids hash 和 prompt template hash 写入 `manifest.json` 的 smoke 字段。
- 已为 `model_compression/generate_teacher_traces.py` 新增多 teacher URL 并发、`--workers`、`--num_shards/--shard_index`、`--resume` 和失败重试能力。
- 已按当前 GPU 约束仅使用 GPU2/GPU3：GPU2 运行 `http://127.0.0.1:8000`，GPU3 运行 `http://127.0.0.1:8001`，并完成 8 条 parallel smoke，两个端点各处理 4 条，`parse_success_rate=1.0`。
- 已为 `model_compression/generate_teacher_traces.py` 新增周期性 checkpoint 和 partial audit：每完成 `--checkpoint_interval` 条就重写当前 trace/distill JSONL 并生成 `.partial.json`，`--resume` 已验证可跳过已完成样本。
- 已完成历史 100 条 G-KD-TRACE broad pilot：GPU2/GPU3 双端点各处理 50 条，`successful_trace_count=100`，`failed_trace_count=0`，`parse_success_rate=1.0`；该结果保留为脚本稳定性证据，不作为第 2 章主实验口径结果。
- 已完成第 2 章主实验 108 条 teacher trace pilot：`GSM8K=36`、`NEU-DET=36`、`CityFlow=36`，GPU2/GPU3 双端点各处理 54 条，`successful_trace_count=108`，`failed_trace_count=0`，`parse_success_rate=1.0`，`action_counts={"pass":87,"inspect":18,"alert":3}`。正式 audit 为 `reports/audit/gate_kd_trace_teacher_chapter2_main_pilot.json`，partial audit 为 `reports/audit/gate_kd_trace_teacher_chapter2_main_pilot.partial.json`。
- 已修复正式全量 teacher trace 长跑中暴露的 NEU-DET mirror 物理目录错位问题：loader 现在按 image/XML stem 在 `NEU-DET/train` 与 `NEU-DET/validation` 中全局配对，不再把项目冻结 split 当作 mirror 物理目录；同时为教师 JSON 响应增加 trailing comma 容错。
- 已完成第 2 章主实验全量 teacher trace gate：`selected_sample_count=8769`，`dataset_counts={"cityflow":36,"gsm8k":7473,"neu_det":1260}`，`successful_trace_count=8769`，`failed_trace_count=0`，`parse_success_rate=1.0`，`action_counts={"alert":76,"inspect":471,"pass":8222}`。正式 audit 为 `reports/audit/gate_kd_trace_teacher.json`，completed partial audit 为 `reports/audit/gate_kd_trace_teacher.partial.json`。
- 已生成 `data/distill/teacher_decision_trace.jsonl` 和 `data/distill/distill_dataset.jsonl`；`teacher_trace_hash=f6f9ce25ffc8aff2cce62978bb1c4b7844a8937a03b0cd5ba137e1d7c0327486`，`distill_dataset_hash=610d23903a5d8d4117c03159268ce068620fee2b97de7ed7ce2dc9fb178c9443`，并写入 `manifest.json`。
- 已新增 `model_compression/run_student_probe.py`，支持 dry-run replay smoke 和 OpenAI-compatible student endpoint，用于生成 `student_probe_trace`、结构化解析结果、teacher/student decision agreement、repair candidate reasons 与审计 hash。
- 已新增 `model_compression/mine_counterfactual_repairs.py`，从 `student_probe_trace` 自动挖掘 action/risk/confidence/review_intent 边界样本并生成 `counterfactual_repair_trace`。
- 已完成 96 条三数据集轮转 Student-Base probe smoke：`dataset_counts={"cityflow":32,"gsm8k":32,"neu_det":32}`，`successful_probe_count=96`，`failed_probe_count=0`，`parse_success_rate=1.0`，`repair_candidate_count=26`，`action_match_rate=0.875`。审计为 `reports/audit/gate_kd_student_probe_smoke.json`，smoke trace hash 为 `8a34e4737c7791e90d4bbb309ea9ac080ba3165c01f3abc7fc418fe97c46b0a3`。
- 已完成 counterfactual repair mining smoke：从 96 条 probe 中生成 26 条 repair trace，`dataset_counts={"cityflow":12,"gsm8k":4,"neu_det":10}`，repair trace hash 为 `c3cb8720062c69eafe12c91eff05343d4e90f49ddb6e41d4c78511b191b9c2df`。审计为 `reports/audit/gate_kd_repair_mining_smoke.json`。

## 未开始

- 1.5B CEDD-Structured adapter 训练、正式 student probe、CEDD-Repair adapter 训练和 INT4 量化。
- GSM8K、HumanEval、CMMLU 的 Cloud/Edge 能力评测脚本与能力保持率汇总。
- P0-B、P1、P2 实验运行。

## 第 2 章当前实验命令

已执行的主实验 pilot 命令：

```bash
python3 model_compression/generate_teacher_traces.py \
  --teacher_url http://127.0.0.1:8000 \
  --teacher_url http://127.0.0.1:8001 \
  --workers 2 \
  --resume \
  --dataset gsm8k \
  --dataset neu_det \
  --dataset cityflow \
  --sample_limit 108 \
  --checkpoint_interval 12 \
  --output_teacher_trace data/distill/teacher_decision_trace.chapter2_main_pilot.jsonl \
  --output_distill data/distill/distill_dataset.chapter2_main_pilot.jsonl \
  --audit reports/audit/gate_kd_trace_teacher_chapter2_main_pilot.json
```

全量主实验 trace 使用 4 个本地 14B-AWQ vLLM endpoint 补跑并通过；修复后同一命令带 `--resume` 已验证会跳过已完成的 8769 条，不重复调用 teacher：

```bash
python3 model_compression/generate_teacher_traces.py \
  --teacher_url http://127.0.0.1:8000 \
  --teacher_url http://127.0.0.1:8001 \
  --teacher_url http://127.0.0.1:8002 \
  --teacher_url http://127.0.0.1:8003 \
  --workers 8 \
  --resume \
  --dataset gsm8k \
  --dataset neu_det \
  --dataset cityflow \
  --checkpoint_interval 50 \
  --output_teacher_trace data/distill/teacher_decision_trace.jsonl \
  --output_distill data/distill/distill_dataset.jsonl \
  --audit reports/audit/gate_kd_trace_teacher.json
```

## 当前可执行检查

```bash
python3 scripts/validate_project_structure.py
python3 scripts/validate_dataset_presence.py
python3 scripts/inspect_datasets.py
python3 scripts/validate_splits.py --check-leakage
bash scripts/setup_datasets.sh --check-only
python3 scripts/build_conflict_gt.py
python3 scripts/generate_formal_manifests.py
python3 scripts/generate_manifest_template.py --overwrite
python3 scripts/validate_manifest_files.py --strict-gdata
python3 scripts/preflight_runtime_smoke.py
python3 scripts/verify_gate_db.py --offline-schema-check
docker compose -f docker/docker-compose.kwdb.yml up -d
python3 scripts/verify_gate_db.py
python3 scripts/download_models.py
python3 scripts/verify_gate_cloud.py --offline-model-check
python3 scripts/serve_vllm_teachers.py --gpu 2 --port 8000
python3 scripts/verify_gate_cloud.py --base-url http://127.0.0.1:8000
python3 model_compression/generate_teacher_traces.py --teacher_url http://127.0.0.1:8000 --sample_limit 32 --output_teacher_trace data/distill/teacher_decision_trace.smoke.jsonl --output_distill data/distill/distill_dataset.smoke.jsonl --audit reports/audit/gate_kd_trace_teacher_smoke.json
python3 scripts/serve_vllm_teachers.py --gpu 2 --gpu 3 --port 8000 --port 8001
python3 model_compression/generate_teacher_traces.py --teacher_url http://127.0.0.1:8000 --teacher_url http://127.0.0.1:8001 --workers 2 --sample_limit 8 --output_teacher_trace data/distill/teacher_decision_trace.parallel_smoke.jsonl --output_distill data/distill/distill_dataset.parallel_smoke.jsonl --audit reports/audit/gate_kd_trace_teacher_parallel_smoke.json
python3 model_compression/generate_teacher_traces.py --teacher_url http://127.0.0.1:8000 --teacher_url http://127.0.0.1:8001 --workers 2 --resume --sample_limit 100 --checkpoint_interval 10 --output_teacher_trace data/distill/teacher_decision_trace.pilot.jsonl --output_distill data/distill/distill_dataset.pilot.jsonl --audit reports/audit/gate_kd_trace_teacher_pilot.json
python3 model_compression/generate_teacher_traces.py --teacher_url http://127.0.0.1:8000 --teacher_url http://127.0.0.1:8001 --workers 2 --resume --dataset gsm8k --dataset neu_det --dataset cityflow --sample_limit 108 --checkpoint_interval 12 --output_teacher_trace data/distill/teacher_decision_trace.chapter2_main_pilot.jsonl --output_distill data/distill/distill_dataset.chapter2_main_pilot.jsonl --audit reports/audit/gate_kd_trace_teacher_chapter2_main_pilot.json
python3 scripts/serve_vllm_teachers.py --gpu 0 --gpu 1 --gpu 2 --gpu 3 --port 8000 --port 8001 --port 8002 --port 8003
python3 model_compression/generate_teacher_traces.py --teacher_url http://127.0.0.1:8000 --teacher_url http://127.0.0.1:8001 --teacher_url http://127.0.0.1:8002 --teacher_url http://127.0.0.1:8003 --workers 8 --resume --dataset gsm8k --dataset neu_det --dataset cityflow --checkpoint_interval 50 --output_teacher_trace data/distill/teacher_decision_trace.jsonl --output_distill data/distill/distill_dataset.jsonl --audit reports/audit/gate_kd_trace_teacher.json
python3 model_compression/run_student_probe.py --dry-run --dataset cityflow --dataset gsm8k --dataset neu_det --sample_limit 96 --output_probe data/distill/student_probe_trace.smoke.jsonl --audit reports/audit/gate_kd_student_probe_smoke.json
python3 model_compression/mine_counterfactual_repairs.py --probe_trace data/distill/student_probe_trace.smoke.jsonl --output_repair data/distill/counterfactual_repair_trace.smoke.jsonl --audit reports/audit/gate_kd_repair_mining_smoke.json --min_repairs 1
```

第一条命令用于检查当前项目骨架、关键目录、基础配置、文档和 manifest 模板是否齐全。第二条命令用于确认 8 个目标数据集目录都已有 payload。第三条命令用于扫描本地数据集目录并生成 `reports/preflight/data_inventory.json`。第四条命令用于验证 split 文件 hash 与 train/validation/test 泄漏检查。第五条命令用于在 Bash 环境检查本地数据集标准目录。第六条命令用于基于冻结 split 重新生成 conflict ground truth manifest 和 audit 文件。第七条命令用于基于冻结 split 重新生成正式 manifest。第八条命令用于按当前脚本重新生成三个 manifest 模板。第九条命令用于严格验收正式 manifest。第十条命令用于执行当前工程骨架的 runtime smoke，并生成 `reports/preflight/runtime_smoke.json`。第十一条命令用于在没有 KWDB 容器时先检查 G-DB schema 文件完整性。第十二条和第十三条命令用于启动 KWDB/KaiwuDB 并执行 G-DB live gate。最后几条命令用于执行 G-CLOUD 离线模型审计、以前台 Ctrl+C 可停止的方式启动 14B-AWQ vLLM teacher 服务、运行 live gate、执行 teacher smoke/parallel smoke/pilot、第 2 章主实验全量 teacher trace gate、student probe smoke 和 repair mining smoke。
