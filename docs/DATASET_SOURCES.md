# Dataset Sources

本文件记录 DB4AI-EdgeServe 的目标数据集、用途、放置目录和接入检查规则。

当前文件只根据 `IMPLEMENTATION_PLAN.md` 固化项目需要的数据集和实验用途；具体下载 URL、账号要求、镜像来源和 license 信息需要在正式下载前逐项核验后补充。不要把未经核验的链接写入这里。

## 通用规则

- 所有真实数据放在 `data/datasets/<dataset_key>/` 下。
- `.gitkeep` 只用于保留空目录，不算真实数据。
- 下载或挂载数据后，先运行 `python scripts/validate_dataset_presence.py` 检查目录是否有 payload。
- 然后运行 `python scripts/inspect_datasets.py` 更新 `reports/preflight/data_inventory.json`。
- 最后由 G-DATA 脚本生成正式 `dataset_manifest.json`、`manifest.json`、`conflict_gt_manifest.json`。
- 数据集版本、样本数量、split hash、sample ids hash 必须写入正式 manifest。

## 数据集清单

| Dataset | Key | Directory | Purpose | Distill | Final Gate | Source Status |
|---|---|---|---|---|---|---|
| GSM8K | `gsm8k` | `data/datasets/gsm8k` | 数学能力评测和 train 蒸馏来源 | train | yes, 500 final samples | source URL to verify |
| HumanEval | `humaneval` | `data/datasets/humaneval` | 代码能力 Final Gate 评测 | no | yes, 164 full | source URL to verify |
| MMLU | `mmlu` | `data/datasets/mmlu` | 英文 NLP 能力评测和 train 蒸馏来源 | train | yes, 1000 stratified samples | source URL to verify |
| CMMLU | `cmmlu` | `data/datasets/cmmlu` | 中文 NLP 能力评测，非 final train 或 synthetic 用于蒸馏 | non-final train or synthetic | yes, 1000 stratified samples | source URL to verify |
| MVTec AD | `mvtec_ad` | `data/datasets/mvtec_ad` | 工业缺陷检测支撑和 Final Gate | train normal only | yes, official test full | source URL to verify |
| NEU-DET | `neu_det` | `data/datasets/neu_det` | 工业缺陷检测辅助数据 | train 70% | yes, 360 stratified samples | source URL to verify |
| CityFlow | `cityflow` | `data/datasets/cityflow` | 交通多摄像头关联、冲突构造和 G6/G7 主评测 | train scenes | yes, script-counted final test | source URL to verify |
| UA-DETRAC | `ua_detrac` | `data/datasets/ua_detrac` | 交通检测/跟踪辅助评测 | no graph training | yes, script-counted final test | source URL to verify |

## 预期接入状态

### GSM8K

- 目录：`data/datasets/gsm8k`
- 需要记录：dataset version、train/test 样本数、final 500 sample ids hash、split hash。
- 注意：final test 子集不得进入蒸馏、planner、policy、graph 或 calibration。

### HumanEval

- 目录：`data/datasets/humaneval`
- 需要记录：dataset version、164 题全量 sample ids hash、执行沙箱设置。
- 注意：不进入蒸馏训练；Final Gate 使用统一 parser 和 timeout。

### MMLU

- 目录：`data/datasets/mmlu`
- 需要记录：dataset version、科目分层策略、final 1000 sample ids hash、split hash。
- 注意：Edge 与 Cloud 必须使用相同 prompt、parser 和采样参数。

### CMMLU

- 目录：`data/datasets/cmmlu`
- 需要记录：dataset version、final 1000 sample ids hash、synthetic 数据 hash。
- 注意：final evaluation 子集不得进入蒸馏；synthetic 需记录 prompt hash、generation seed、teacher model 和 sample hash。

### MVTec AD

- 目录：`data/datasets/mvtec_ad`
- 需要记录：dataset version、train normal 样本数、official test 是否全量进入 Final Gate。
- 注意：validation 使用 train 正常子集、NEU-DET val 缺陷样本和非 final 派生异常样本，不使用官方 test 做 validation。

### NEU-DET

- 目录：`data/datasets/neu_det`
- 需要记录：dataset version、每类 train/validation/test 数量、类别分层 split hash。
- 注意：计划中 final test 为 360，train 使用 70%。

### CityFlow

- 目录：`data/datasets/cityflow`
- 需要记录：dataset version、camera/intersection 子集、vehicle id 数、relation edge/group/conflict group 统计。
- 注意：优先使用 CityFlow-Original；若实际为 CityFlowV2，必须在 `dataset_manifest.json` 记录实际版本。若官方 test 标签不可用，按 scene/camera/time window 冻结划分并记录 split hash。

### UA-DETRAC

- 目录：`data/datasets/ua_detrac`
- 需要记录：dataset version、视频/帧/标注统计、final sample ids hash。
- 注意：仅声明同视频内 track id 或 derived tracklet，不把同视频轨迹人工外推为跨摄像头 identity。

## 接入后检查

```powershell
python scripts/validate_dataset_presence.py
python scripts/inspect_datasets.py
```

当前 P0-A0 阶段如果数据尚未下载，可以使用：

```powershell
python scripts/validate_dataset_presence.py --allow-empty
```

正式进入 G-DATA 后，不应使用 `--allow-empty` 作为通过依据。
