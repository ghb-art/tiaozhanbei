# Project Status

更新时间：2026-07-01

当前阶段：G-DATA 数据接入，已完成 8 个目标数据集的本地 payload 挂载和 preflight 盘点。

## 当前结论

项目已经有完整的研究方案、目录规划和本地数据 payload。当前尚未完成 split 冻结、正式 manifest 生成、模型训练、数据库部署或实验运行。

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

## 未开始

- 数据集 split 冻结。
- MVTec AD、CityFlow、UA-DETRAC 大型归档包展开或脚本化读取。
- 正式 `dataset_manifest.json`、`manifest.json`、`conflict_gt_manifest.json` 生成。
- KWDB/KaiwuDB Docker 部署与 schema 创建。
- 14B 云端教师模型服务验证。
- 1.5B 学生模型蒸馏、repair 和 INT4 量化。
- P0-B、P1、P2 实验运行。

## 当前可执行检查

```powershell
python scripts/validate_project_structure.py
python scripts/validate_dataset_presence.py
python scripts/inspect_datasets.py
python scripts/generate_manifest_template.py --overwrite
python scripts/validate_manifest_files.py --allow-missing
python scripts/preflight_runtime_smoke.py
```

第一条命令用于检查当前项目骨架、关键目录、基础配置、文档和 manifest 模板是否齐全。第二条命令用于确认 8 个目标数据集目录都已有 payload。第三条命令用于扫描本地数据集目录并生成 `reports/preflight/data_inventory.json`。第四条命令用于按当前脚本重新生成三个 manifest 模板。第五条命令用于在正式 manifest 尚未生成时验证验收脚本可执行；生成正式 manifest 后应改用 `python scripts/validate_manifest_files.py --strict-gdata`。第六条命令用于执行当前工程骨架的 runtime smoke，并生成 `reports/preflight/runtime_smoke.json`。
