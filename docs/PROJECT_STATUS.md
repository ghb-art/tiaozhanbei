# Project Status

更新时间：2026-06-30

当前阶段：P0-A0 工程骨架准备，已完成 G-DATA 前置 manifest 模板规范、数据目录盘点、数据接入规范、数据来源核验和 runtime smoke 预检。

## 当前结论

项目已经有完整的研究方案和目录规划，但尚未进入数据下载、模型训练、数据库部署或实验运行阶段。

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
- 已生成 `reports/preflight/data_inventory.json`。当前 8 个预期数据集目录均存在，但都只有 `.gitkeep`，尚无真实 payload 数据。
- 已新增 `docs/DATASET_SOURCES.md`，记录 8 个目标数据集的用途、放置目录、接入注意事项和待核验来源状态。
- 已核验并补充 `docs/DATASET_SOURCES.md` 中 8 个目标数据集的官方/主来源、访问方式、license/terms 注意事项和下载前人工确认要求。
- 已新增 `scripts/validate_dataset_presence.py`，用于检查数据集目录是否存在 `.gitkeep` 之外的真实 payload 文件。
- 已新增 `scripts/preflight_runtime_smoke.py`，用于检查 Python 版本、关键目录、配置/模板可读性，并调用现有轻量校验脚本完成 runtime smoke。
- 已生成 `reports/preflight/runtime_smoke.json`，作为当前工程骨架可运行性的预检报告。

## 未开始

- 数据集下载与 split 冻结。
- 数据集真实文件接入。
- MVTec AD、NEU-DET、CityFlow、UA-DETRAC 下载前 license、数据协议和账号/表单要求的人工复核记录。
- 正式 `dataset_manifest.json`、`manifest.json`、`conflict_gt_manifest.json` 生成。
- KWDB/KaiwuDB Docker 部署与 schema 创建。
- 14B 云端教师模型服务验证。
- 1.5B 学生模型蒸馏、repair 和 INT4 量化。
- P0-B、P1、P2 实验运行。

## 当前可执行检查

```powershell
python scripts/validate_project_structure.py
python scripts/validate_dataset_presence.py --allow-empty
python scripts/inspect_datasets.py
python scripts/generate_manifest_template.py --overwrite
python scripts/validate_manifest_files.py --allow-missing
python scripts/preflight_runtime_smoke.py
```

第一条命令用于检查当前项目骨架、关键目录、基础配置、文档和 manifest 模板是否齐全。第二条命令用于在数据尚未下载时确认目标目录存在；进入 G-DATA 后应去掉 `--allow-empty`。第三条命令用于扫描本地数据集目录并生成 `reports/preflight/data_inventory.json`。第四条命令用于按当前脚本重新生成三个 manifest 模板。第五条命令用于在正式 manifest 尚未生成时验证验收脚本可执行；进入 G-DATA 后应改用 `python scripts/validate_manifest_files.py --strict-gdata`。第六条命令用于执行当前工程骨架的 runtime smoke，并生成 `reports/preflight/runtime_smoke.json`。
