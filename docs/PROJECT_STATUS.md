# Project Status

更新时间：2026-07-02

当前阶段：G-KD-TRACE 已完成 teacher trace smoke，下一步生成全量 train split 教师 trace 与蒸馏数据。

## 当前结论

项目已经有完整的研究方案、目录规划、本地数据 payload、冻结 split、正式 manifest、conflict audit 文件、可运行的 KWDB/KaiwuDB 单节点验证、通过 live gate 的 14B-AWQ vLLM 云端教师服务验证，以及 32 条 train split 教师结构化决策 trace smoke。下一步应使用同一脚本生成全量 train split 教师 trace 和蒸馏数据集。

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

## 未开始

- 1.5B 学生模型蒸馏、repair 和 INT4 量化。
- G-KD-TRACE 全量 train split 教师结构化决策 trace 与蒸馏数据集生成。
- P0-B、P1、P2 实验运行。

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
CUDA_VISIBLE_DEVICES=0 VLLM_WORKER_MULTIPROC_METHOD=spawn .venv/bin/vllm serve models/pretrained/Qwen--Qwen2.5-14B-Instruct-AWQ --quantization awq --max-model-len 4096 --gpu-memory-utilization 0.85 --tensor-parallel-size 1 --port 8000
python3 scripts/verify_gate_cloud.py --base-url http://127.0.0.1:8000
python3 model_compression/generate_teacher_traces.py --teacher_url http://127.0.0.1:8000 --sample_limit 32 --output_teacher_trace data/distill/teacher_decision_trace.smoke.jsonl --output_distill data/distill/distill_dataset.smoke.jsonl --audit reports/audit/gate_kd_trace_teacher_smoke.json
```

第一条命令用于检查当前项目骨架、关键目录、基础配置、文档和 manifest 模板是否齐全。第二条命令用于确认 8 个目标数据集目录都已有 payload。第三条命令用于扫描本地数据集目录并生成 `reports/preflight/data_inventory.json`。第四条命令用于验证 split 文件 hash 与 train/validation/test 泄漏检查。第五条命令用于在 Bash 环境检查本地数据集标准目录。第六条命令用于基于冻结 split 重新生成 conflict ground truth manifest 和 audit 文件。第七条命令用于基于冻结 split 重新生成正式 manifest。第八条命令用于按当前脚本重新生成三个 manifest 模板。第九条命令用于严格验收正式 manifest。第十条命令用于执行当前工程骨架的 runtime smoke，并生成 `reports/preflight/runtime_smoke.json`。第十一条命令用于在没有 KWDB 容器时先检查 G-DB schema 文件完整性。第十二条和第十三条命令用于启动 KWDB/KaiwuDB 并执行 G-DB live gate。倒数四条命令用于执行 G-CLOUD 离线模型审计、启动 14B-AWQ vLLM 服务、运行 live gate，并执行 32 条 G-KD-TRACE teacher smoke。
