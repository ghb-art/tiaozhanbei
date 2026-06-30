# Project Status

更新时间：2026-06-30

当前阶段：P0-A0 工程骨架准备，已完成 G-DATA 前置 manifest 模板规范和数据目录盘点。

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

## 未开始

- 数据集下载与 split 冻结。
- 数据集真实文件接入。
- 正式 `dataset_manifest.json`、`manifest.json`、`conflict_gt_manifest.json` 生成。
- KWDB/KaiwuDB Docker 部署与 schema 创建。
- 14B 云端教师模型服务验证。
- 1.5B 学生模型蒸馏、repair 和 INT4 量化。
- P0-B、P1、P2 实验运行。

## 当前可执行检查

```powershell
python scripts/validate_project_structure.py
python scripts/inspect_datasets.py
python scripts/generate_manifest_template.py --overwrite
python scripts/validate_manifest_files.py --allow-missing
```

第一条命令用于检查当前项目骨架、关键目录、基础配置、文档和 manifest 模板是否齐全。第二条命令用于扫描本地数据集目录并生成 `reports/preflight/data_inventory.json`。第三条命令用于按当前脚本重新生成三个 manifest 模板。第四条命令用于在正式 manifest 尚未生成时验证验收脚本可执行；进入 G-DATA 后应改用 `python scripts/validate_manifest_files.py --strict-gdata`。
