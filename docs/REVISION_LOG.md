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
