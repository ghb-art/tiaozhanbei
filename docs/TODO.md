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
- [ ] 创建 preflight runtime smoke 脚本。

## G-DATA

- [ ] 下载或挂载 8 个目标数据集到 `data/datasets/`。
- [ ] 重新运行 `scripts/inspect_datasets.py`，确认数据目录不再为空。
- [ ] 基于模板生成正式 `dataset_manifest.json`、`manifest.json`、`conflict_gt_manifest.json`。
- [ ] 使用 `scripts/validate_manifest_files.py --strict-gdata` 验收正式 manifest。
- [ ] 实现 `scripts/setup_datasets.sh`。
- [ ] 实现 `scripts/validate_splits.py`。
- [ ] 实现 `scripts/build_conflict_gt.py`。
- [ ] 生成 `dataset_manifest.json`。
- [ ] 生成 `conflict_gt_manifest.json`。
- [ ] 生成 conflict ground truth audit 文件。

## G-DB

- [ ] 创建 `sql/cloud_schema.sql`。
- [ ] 创建 `docker/docker-compose.kwdb.yml`。
- [ ] 实现 `scripts/verify_gate_db.py`。

## G-CLOUD

- [ ] 确认 14B-AWQ 服务启动方式。
- [ ] 实现 `scripts/verify_gate_cloud.py`。
- [ ] 记录 model hash 和 prompt hash。
