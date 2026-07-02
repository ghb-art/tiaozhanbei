# TODO

本文件用于把 `IMPLEMENTATION_PLAN.md` 中的大阶段拆成可执行任务。

## P0-A0 工程骨架

- [x] 初始化 Git 仓库。
- [x] 建立基础 `.gitignore`。
- [x] 创建 `docs/REVISION_LOG.md`。
- [x] 创建 `docs/PROJECT_STATUS.md`。
- [x] 创建 `docs/TODO.md`。
- [x] 创建网络与工作负载 profile 配置。
- [x] 创建模型与 dev 配置。
- [x] 创建项目结构校验脚本。
- [x] 创建 manifest 模板生成脚本。
- [x] 生成 `dataset_manifest.template.json`、`manifest.template.json`、`conflict_gt_manifest.template.json`。
- [x] 更新结构校验脚本，检查 manifest 模板是否存在且 JSON 可解析。
- [x] 创建正式 manifest 验收脚本 `scripts/validate_manifest_files.py`。
- [x] 创建数据集目录盘点脚本 `scripts/inspect_datasets.py`。
- [x] 生成 `reports/preflight/data_inventory.json`。
- [x] 创建 `docs/DATASET_SOURCES.md`。
- [x] 创建数据集 payload 存在性校验脚本 `scripts/validate_dataset_presence.py`。
- [x] 创建 preflight runtime smoke 脚本。

## G-DATA

- [x] 核验并补充 `docs/DATASET_SOURCES.md` 中每个数据集的正式下载来源、license 和账号要求。
- [x] 记录 MVTec AD、NEU-DET、CityFlow、UA-DETRAC 的 license/数据协议或 mirror fallback 状态。
- [x] 下载或挂载 8 个目标数据集到 `data/datasets/`。
- [x] 使用 `scripts/validate_dataset_presence.py` 验收数据集 payload 存在性。
- [x] 重新运行 `scripts/inspect_datasets.py`，确认数据目录不再为空。
- [x] 展开 MVTec AD、CityFlow、UA-DETRAC 大型归档包到标准数据目录。
- [x] 实现 `scripts/validate_splits.py` 并冻结 8 个数据集 split。
- [x] 基于模板生成正式 `dataset_manifest.json`、`manifest.json`、`conflict_gt_manifest.json`。
- [x] 使用 `scripts/validate_manifest_files.py --strict-gdata` 验收正式 manifest。
- [x] 实现 `scripts/setup_datasets.sh`。
- [x] 实现 `scripts/validate_splits.py`。
- [x] 实现 `scripts/build_conflict_gt.py`。
- [x] 生成 `dataset_manifest.json`。
- [x] 生成 `conflict_gt_manifest.json`。
- [x] 生成 conflict ground truth audit 文件。

## G-DB

- [x] 创建 `sql/cloud_schema.sql`。
- [x] 创建 `docker/docker-compose.kwdb.yml`。
- [x] 实现 `scripts/verify_gate_db.py`。
- [x] 启动 KWDB Docker 并运行 `scripts/verify_gate_db.py` 完成 live gate。

## G-CLOUD

- [x] 下载 `Qwen/Qwen2.5-14B-Instruct-AWQ`、`Qwen/Qwen2.5-7B-Instruct-AWQ`、`Qwen/Qwen2.5-1.5B-Instruct` 到本地 `models/pretrained/`。
- [x] 确认 14B-AWQ 服务启动方式。
- [x] 新增前台 vLLM teacher 启动器，统一用 Ctrl+C 停止多 endpoint 服务。
- [x] 实现 `scripts/verify_gate_cloud.py`。
- [x] 记录 model hash 和 prompt hash。
- [x] 通过 14B-AWQ vLLM live gate：`/health` 200，首 token 0.581s。

## G-KD-TRACE

- [x] 实现教师结构化决策 trace 生成脚本。
- [x] 使用 14B-AWQ vLLM 服务完成 32 条 train split teacher trace smoke，并生成 `reports/audit/gate_kd_trace_teacher_smoke.json`。
- [x] 为 teacher trace 生成脚本加入多 teacher URL 并发、分片、断点续跑和重试能力，并用 GPU2/GPU3 完成 8 条 parallel smoke。
- [x] 为 teacher trace 生成脚本加入周期性 checkpoint/partial audit，并用 GPU2/GPU3 完成 100 条 pilot。
- [x] 将第 2 章主实验数据口径固定为 `GSM8K`、`HumanEval`、`CMMLU`、`NEU-DET`、`CityFlow`，并将 `MMLU`、`MVTec AD`、`UA-DETRAC` 标记为支持/备份资产。
- [x] 按第 2 章主实验口径完成 108 条 teacher trace pilot，并生成 `reports/audit/gate_kd_trace_teacher_chapter2_main_pilot.json`。
- [x] 基于第 2 章主实验 train split 生成 `data/distill/teacher_decision_trace.jsonl`。
- [x] 生成 `data/distill/distill_dataset.jsonl` 并记录 hash。
- [x] 实现 Student-Base probe trace 脚本，并完成 96 条三数据集轮转 smoke。
- [x] 实现 counterfactual repair mining 脚本，并完成 repair trace smoke。
- [ ] 实现 GSM8K、HumanEval、CMMLU 的 Cloud/Edge 能力评测脚本和能力保持率汇总。
- [ ] 训练 CEDD-Structured adapter 并运行正式 train split student probe。
- [ ] 生成正式 `data/distill/student_probe_trace.jsonl` 并记录 hash。
- [ ] 生成正式 `data/distill/counterfactual_repair_trace.jsonl` 并记录 hash。
- [ ] 训练 CEDD-Repair adapter 并生成 INT4 量化行为 trace。
