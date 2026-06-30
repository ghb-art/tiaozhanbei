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
