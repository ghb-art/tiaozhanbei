# Revision Log

本文件记录 DB4AI-EdgeServe 项目实施过程中的方案调整、工程 fallback、数据口径变化、脚本修复和 hash 影响。

所有会影响 `dataset_manifest.json`、`manifest.json`、`conflict_gt_manifest.json`、模型产物、指标脚本或 Final Gate 口径的变更，都必须在这里留下记录。

## 记录规则

| 字段 | 含义 |
|---|---|
| Date | 变更发生日期，使用北京时间自然日 |
| Stage | 所属阶段，如 P0-A0、G-DATA、G-DB、P1、P2 |
| Change | 变更内容 |
| Reason | 为什么需要变更 |
| Hash Impact | 是否影响数据、模型、配置或指标 hash |
| Artifacts | 相关文件或产物 |

## Entries

| Date | Stage | Change | Reason | Hash Impact | Artifacts |
|---|---|---|---|---|---|
| 2026-06-30 | P0-A0 | 初始化工程管理文档、基础配置和结构校验脚本 | 将方案文档推进到可检查的工程骨架 | No Final hash yet | `docs/`, `configs/`, `scripts/validate_project_structure.py` |
| 2026-06-30 | P0-A0 | 新增 manifest 模板生成脚本和三类 template JSON | 按实施方案固化 G-DATA/Final Gate 产物字段标准 | No Final hash yet | `scripts/generate_manifest_template.py`, `*.template.json` |
| 2026-06-30 | P0-A0 | 新增正式 manifest 验收脚本 | 在数据接入前自动化 G-DATA/Final Gate manifest 基础验收规则 | No Final hash yet | `scripts/validate_manifest_files.py`, `scripts/validate_project_structure.py` |
| 2026-06-30 | P0-A0 | 新增数据集目录盘点脚本并生成 preflight inventory | 在正式 G-DATA 前确认本地数据目录现状 | No Final hash yet | `scripts/inspect_datasets.py`, `reports/preflight/data_inventory.json` |
| 2026-06-30 | P0-A0 | 新增数据来源说明和数据集 payload 存在性校验脚本 | 在下载/挂载数据前固定数据接入规则并避免 `.gitkeep` 被误判为真实数据 | No Final hash yet | `docs/DATASET_SOURCES.md`, `scripts/validate_dataset_presence.py` |
| 2026-06-30 | G-DATA | 核验并补充 8 个目标数据集的官方/主来源、访问方式和 license/terms 注意事项 | 在正式下载数据前固定来源口径，降低使用未授权镜像或混入 final 数据的风险 | No Final hash yet | `docs/DATASET_SOURCES.md`, `docs/TODO.md`, `docs/PROJECT_STATUS.md` |
| 2026-06-30 | P0-A0 | 新增 runtime smoke 预检脚本并生成预检报告 | 在进入数据下载和模型运行前确认工程骨架、配置、模板和轻量校验脚本可执行 | No Final hash yet | `scripts/preflight_runtime_smoke.py`, `reports/preflight/runtime_smoke.json`, `scripts/validate_project_structure.py` |
| 2026-07-01 | G-DATA | 接入 8 个目标数据集 payload，并记录 NEU-DET/UA-DETRAC Kaggle mirror fallback | 官方 NEU-DET 页面不可达、UA-DETRAC 官方页未提供直接下载入口；先以本地结构校验和 archive hash 固定 fallback 口径 | No Final hash yet; archive hashes recorded | `data/datasets/`, `reports/preflight/data_inventory.json`, `docs/DATASET_SOURCES.md`, `.gitignore` |
| 2026-07-01 | G-DATA | 展开大型数据集、冻结 split、修复 NEU-DET split 错位并生成正式 manifest | 将 preflight 数据接入推进到可严格验收的 G-DATA manifest 状态 | Yes, split and manifest hashes generated | `data/splits/`, `dataset_manifest.json`, `manifest.json`, `conflict_gt_manifest.json`, `scripts/validate_splits.py`, `scripts/generate_formal_manifests.py` |
| 2026-07-01 | G-DATA | 新增数据集本地检查脚本和 conflict GT 独立生成脚本 | 补齐实施计划中的 `setup_datasets.sh` 与 `build_conflict_gt.py`，让数据集布局检查和冲突 GT 生成可单独执行、可复验 | Yes, manifest hashes regenerated | `scripts/setup_datasets.sh`, `scripts/build_conflict_gt.py`, `scripts/generate_formal_manifests.py`, `conflict_gt_manifest.json`, `manifest.json` |
| 2026-07-01 | G-DATA/G-DB | 生成 conflict GT 审计文件，并新增 KWDB schema、compose 与 G-DB 验证脚本 | 闭合 G-DATA audit 产物要求，同时推进下一阶段数据库 gate 的可执行骨架 | Yes, audit and manifest hashes regenerated | `reports/audit/conflict_gt_audit.csv`, `reports/audit/conflict_gt_sample_audit.json`, `reports/audit/gate_db_schema_check.json`, `sql/cloud_schema.sql`, `docker/docker-compose.kwdb.yml`, `scripts/verify_gate_db.py`, `manifest.json` |
| 2026-07-01 | G-DB | 启动 KWDB Docker 容器并完成 G-DB live gate | 验证 schema 创建、写入、查询和 CSV 导出链路可运行 | Yes, DB smoke report generated | `docker/docker-compose.kwdb.yml`, `scripts/verify_gate_db.py`, `reports/audit/gate_db_smoke.csv` |
| 2026-07-01 | G-CLOUD | 下载三份 Qwen 模型到本地预训练模型目录并生成下载审计 | 为后续上传服务器和启动 14B-AWQ 云端教师服务做准备 | Yes, local model audit generated | `scripts/download_models.py`, `models/pretrained/`, `reports/audit/model_downloads.json` |
| 2026-07-01 | G-CLOUD | 新增云端教师服务验证脚本并完成 14B-AWQ vLLM live gate | 闭合实施计划中 `/health` 200、smoke 首 token <2s、model_hash/prompt_hash 记录要求；服务器缺少 vLLM，已在项目 `.venv` 中隔离安装 `vllm==0.8.5` 并固定 `transformers==4.51.3` 规避 Qwen tokenizer API 不兼容 | Yes, cloud gate audit and manifest hashes updated | `scripts/verify_gate_cloud.py`, `reports/audit/gate_cloud_smoke.json`, `manifest.json`, `docs/TODO.md`, `docs/PROJECT_STATUS.md` |
| 2026-07-02 | G-KD-TRACE | 新增教师结构化决策 trace 生成脚本并完成 32 条 train split smoke | 修复缺失的 `model_compression/generate_teacher_traces.py`，打通 frozen split 采样、数据集 loader、OpenAI-compatible vLLM 调用、结构化 JSON 解析、distill JSONL 和审计输出链路 | Yes, smoke trace/distill/audit hashes recorded; final full-train hashes not replaced | `model_compression/generate_teacher_traces.py`, `reports/audit/gate_kd_trace_teacher_smoke.json`, `manifest.json`, `docs/TODO.md`, `docs/PROJECT_STATUS.md` |
| 2026-07-02 | G-KD-TRACE | teacher trace 生成脚本新增多端点并发、分片、断点续跑和重试，并完成 GPU2/GPU3 parallel smoke | 后续全量 train trace 规模较大，需在不使用 GPU0/GPU1 的约束下并行利用 GPU2/GPU3 两个 14B-AWQ teacher 副本提升吞吐并支持中断恢复 | Yes, parallel smoke audit hashes recorded; final full-train hashes not replaced | `model_compression/generate_teacher_traces.py`, `reports/audit/gate_kd_trace_teacher_parallel_smoke.json`, `manifest.json`, `docs/TODO.md`, `docs/PROJECT_STATUS.md` |
| 2026-07-02 | G-KD-TRACE | teacher trace 生成脚本新增周期性 checkpoint/partial audit，并完成 100 条 GPU2/GPU3 pilot | 全量 train trace 长跑需要可恢复落盘；pilot 验证 checkpoint、resume、双端点负载均衡和结构化解析稳定性 | Yes, pilot trace/distill/audit hashes recorded; final full-train hashes not replaced | `model_compression/generate_teacher_traces.py`, `reports/audit/gate_kd_trace_teacher_pilot.json`, `reports/audit/gate_kd_trace_teacher_pilot.partial.json`, `manifest.json`, `docs/TODO.md`, `docs/PROJECT_STATUS.md` |
| 2026-07-02 | G-KD-TRACE / Chapter 2 | 将第 2 章主实验口径收敛为 GSM8K、HumanEval、CMMLU、NEU-DET、CityFlow，并完成 108 条主口径 teacher trace pilot | 当前论文实验安排要求 MMLU、MVTec AD、UA-DETRAC 不进入主实验；teacher trace 训练数据只使用有 train split 的 GSM8K、NEU-DET、CityFlow，HumanEval/CMMLU 保留为能力评测集 | Yes, chapter2 main pilot audit/trace/distill hashes recorded; historical broad pilot retained as stability evidence | `IMPLEMENTATION_PLAN.md`, `dataset_manifest.json`, `manifest.json`, `scripts/generate_formal_manifests.py`, `model_compression/generate_teacher_traces.py`, `reports/audit/gate_kd_trace_teacher_chapter2_main_pilot.json`, `reports/audit/gate_kd_trace_teacher_chapter2_main_pilot.partial.json`, `docs/TODO.md`, `docs/PROJECT_STATUS.md`, `docs/DATASET_SOURCES.md`, `docs/DATASET_SPLIT_STRATEGY.md` |
| 2026-07-02 | G-KD-TRACE / Chapter 2 | 修复 NEU-DET teacher trace loader 的 mirror 物理目录错位和教师 JSON trailing comma 容错，并完成 8769 条全量 train split teacher trace gate | frozen split 中的 `neu_det/train/*` 是项目划分，不等于 Kaggle mirror 物理 `train/validation` 目录；正式长跑暴露 240+ stem 图像/XML 需跨 mirror 目录全局配对，同时一条 GSM8K 教师响应需要轻量 JSON 容错 | Yes, full teacher trace/distill/audit hashes recorded and manifest regenerated | `model_compression/generate_teacher_traces.py`, `scripts/generate_formal_manifests.py`, `reports/audit/gate_kd_trace_teacher.json`, `reports/audit/gate_kd_trace_teacher.partial.json`, `manifest.json`, `dataset_manifest.json`, `conflict_gt_manifest.json`, `reports/audit/conflict_gt_audit.csv`, `reports/audit/conflict_gt_sample_audit.json`, `docs/TODO.md`, `docs/PROJECT_STATUS.md` |
| 2026-07-02 | G-KD-TRACE / CEDD | 新增 Student-Base probe 与 counterfactual repair mining 脚本，并完成 96 条 probe smoke 和 26 条 repair smoke | 进入 CEDD 学生在环阶段前，需要先固定 student_probe_trace 与 counterfactual_repair_trace 的 schema、候选挖掘规则、hash 和审计输出；当前 GPU 被 14B teacher 服务占用，本次只闭合 dry-run smoke，不声明正式学生训练完成 | Yes, smoke audit hashes recorded and manifest regenerated; final student/repair hashes not replaced | `model_compression/run_student_probe.py`, `model_compression/mine_counterfactual_repairs.py`, `reports/audit/gate_kd_student_probe_smoke.json`, `reports/audit/gate_kd_repair_mining_smoke.json`, `scripts/generate_formal_manifests.py`, `scripts/validate_project_structure.py`, `scripts/preflight_runtime_smoke.py`, `manifest.json`, `docs/TODO.md`, `docs/PROJECT_STATUS.md` |
| 2026-07-02 | G-CLOUD/G-KD-TRACE | 新增前台 vLLM teacher 启动器，替代后续文档中的 `systemd-run` teacher 启动方式 | 14B teacher 端点此前混合使用 VS Code terminal 和 user systemd unit，停止方式不一致；统一为单个前台 launcher 后，可通过 Ctrl+C 优雅停止所有子进程组 | No model/data hash change; runtime command and docs updated | `scripts/serve_vllm_teachers.py`, `scripts/validate_project_structure.py`, `scripts/preflight_runtime_smoke.py`, `docs/PROJECT_STATUS.md`, `docs/TODO.md` |

## Fallback Event Template

```text
Date:
Stage:
Fallback type: data_path | field_mapping | script_param | runtime_recovery | engineering_fix
Description:
Reason:
Allowed by plan: yes/no
Affected artifacts:
Hash regeneration required: yes/no
Validation command:
Result:
```
