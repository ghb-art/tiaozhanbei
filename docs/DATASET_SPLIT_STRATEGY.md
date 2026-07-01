# Dataset Split Strategy

本文件记录 G-DATA 阶段的固定划分方法。所有划分使用 `sampling_seed=42`，输出由 `scripts/validate_splits.py --write --check-leakage` 生成到 `data/splits/`。

## 通用原则

- Final Gate 的 test 子集不得进入 train、validation、distill、planner、policy、graph 或 calibration。
- 每个 split 的 sample id 文件必须固定，并由 `frozen_splits.json` 记录 count、hash 和全局 `global_split_hash`。
- Kaggle mirror 只作为 fallback source，正式 manifest 必须记录 mirror URL、archive hash 和本地结构校验结果。

## 数据集划分

| Dataset | Train | Validation | Final Test | Method |
|---|---|---|---|---|
| GSM8K | official train 全量 | none | official test 中 seed=42 固定 500 条 | 保持 OpenAI 官方 train/test 边界。 |
| HumanEval | none | none | official 164 tasks 全量 | 只用于代码能力 Final Gate。 |
| MMLU | auxiliary_train + official dev | official val | official test 分科目 stratified 1000 条 | 按科目比例采样，固定 sample ids。 |
| CMMLU | none | official dev | official test 分科目 stratified 1000 条 | 中文 Final Gate 不进入蒸馏训练。 |
| MVTec AD | official train/good 全量 | none | official test 全量 | official test 不用于 validation。 |
| NEU-DET | 每类 210 | 每类 30 | 每类 60 | 不沿用 mirror split，先按 image/xml stem 配对，再做 70/10/20 分层划分。 |
| CityFlow | official train camera dirs | official validation camera dirs | official test camera dirs | 关系统计来自 AI City eval ground truth。 |
| UA-DETRAC | none | train XML sequences | test XML sequences | 镜像保留官方风格 60/40 annotation split；不用于 graph training。 |

## NEU-DET 特别处理

Kaggle mirror 中 `crazing_240.jpg` 位于 train，而 `crazing_240.xml` 位于 validation。冻结 split 时不使用 mirror 目录划分，而是扫描全量图片和 XML，按文件 stem 重新配对后再分层采样，因此该错位不会进入正式 split。
