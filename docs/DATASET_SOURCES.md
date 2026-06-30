# Dataset Sources

本文件记录 DB4AI-EdgeServe 的目标数据集、用途、放置目录、官方或主来源、访问方式和接入检查规则。

数据源核验日期：2026-06-30。

当前状态：已完成来源核验，但尚未下载或挂载真实数据。后续下载数据前，仍需人工确认对应页面上的最新 license、数据协议、账号要求和下载范围；真实数据不得提交进 Git。

## 通用规则

- 所有真实数据放在 `data/datasets/<dataset_key>/` 下。
- `.gitkeep` 只用于保留空目录，不算真实数据。
- 不自动下载大型数据集，也不绕过表单、账号、license agreement 或挑战赛数据协议。
- 下载或挂载数据后，先运行 `python scripts/validate_dataset_presence.py` 检查目录是否有 payload。
- 然后运行 `python scripts/inspect_datasets.py` 更新 `reports/preflight/data_inventory.json`。
- 最后由 G-DATA 脚本生成正式 `dataset_manifest.json`、`manifest.json`、`conflict_gt_manifest.json`。
- 数据集版本、来源 URL、下载日期、license/terms、样本数量、split hash、sample ids hash 必须写入正式 manifest。

## 核验状态说明

| Status | 含义 |
|---|---|
| `verified-primary` | 已定位到官方仓库、作者仓库或官方数据页，可作为主来源。 |
| `verified-primary-manual-download` | 已定位到官方主来源，但下载需要人工点击、填表、登录或接受数据协议。 |
| `source-located-license-pending` | 已定位到主来源，但页面访问或 license 信息仍需下载前人工复核。 |

## 数据集清单

| Dataset | Key | Directory | Purpose | Distill | Final Gate | Source | Access / License | Status |
|---|---|---|---|---|---|---|---|---|
| GSM8K | `gsm8k` | `data/datasets/gsm8k` | 数学能力评测和 train 蒸馏来源 | train | yes, 500 final samples | [OpenAI grade-school-math][gsm8k-source] | public GitHub; repo MIT license | `verified-primary` |
| HumanEval | `humaneval` | `data/datasets/humaneval` | 代码能力 Final Gate 评测 | no | yes, 164 full | [OpenAI human-eval][humaneval-source] | public GitHub; repo MIT license; 执行评测需沙箱 | `verified-primary` |
| MMLU | `mmlu` | `data/datasets/mmlu` | 英文 NLP 能力评测和 train 蒸馏来源 | train | yes, 1000 stratified samples | [hendrycks/test][mmlu-source] | public GitHub; repo MIT license; README 链接 data tar | `verified-primary` |
| CMMLU | `cmmlu` | `data/datasets/cmmlu` | 中文 NLP 能力评测，非 final train 或 synthetic 用于蒸馏 | non-final train or synthetic | yes, 1000 stratified samples | [haonan-li/CMMLU][cmmlu-source] / [HF mirror][cmmlu-hf] | repo states CC BY-NC-SA 4.0; HF metadata shows CC BY-NC | `verified-primary` |
| MVTec AD | `mvtec_ad` | `data/datasets/mvtec_ad` | 工业缺陷检测支撑和 Final Gate | train normal only | yes, official test full | [MVTec official dataset page][mvtec-source] | official page form download; CC BY-NC-SA 4.0; non-commercial only | `verified-primary-manual-download` |
| NEU-DET | `neu_det` | `data/datasets/neu_det` | 工业缺陷检测辅助数据 | train 70% | yes, 360 stratified samples | [NEU faculty page][neu-source] | official faculty page located; license/terms must be manually checked before use | `source-located-license-pending` |
| CityFlow | `cityflow` | `data/datasets/cityflow` | 交通多摄像头关联、冲突构造和 G6/G7 主评测 | train scenes | yes, script-counted final test | [AI City 2022 Track 1][cityflow-source] | download link accepts AI City data license agreement; manual download required | `verified-primary-manual-download` |
| UA-DETRAC | `ua_detrac` | `data/datasets/ua_detrac` | 交通检测/跟踪辅助评测 | no graph training | yes, script-counted final test | [UAlbany/CVML project page][uadetrac-source] / [UBMDFL downloads][uadetrac-downloads] | research-use download page; license/terms must be manually checked before use | `verified-primary-manual-download` |

## 逐项接入记录

### GSM8K

- 目录：`data/datasets/gsm8k`
- 主来源：OpenAI `grade-school-math` 仓库。
- 下载方式：从 GitHub 仓库获取 `grade_school_math/data/train.jsonl`、`grade_school_math/data/test.jsonl` 和需要的 Socratic 文件。
- 需要记录：commit hash 或下载日期、dataset version、train/test 样本数、final 500 sample ids hash、split hash。
- 注意：final test 子集不得进入蒸馏、planner、policy、graph 或 calibration。

### HumanEval

- 目录：`data/datasets/humaneval`
- 主来源：OpenAI `human-eval` 仓库。
- 下载方式：从 GitHub 仓库获取 `data` 与 `human_eval` 相关评测文件。
- 需要记录：commit hash 或下载日期、dataset version、164 题全量 sample ids hash、执行沙箱设置。
- 注意：不进入蒸馏训练；Final Gate 使用统一 parser 和 timeout；运行模型生成代码时必须使用隔离沙箱。

### MMLU

- 目录：`data/datasets/mmlu`
- 主来源：Hendrycks `test` 仓库。
- 下载方式：仓库 README 指向 `https://people.eecs.berkeley.edu/~hendrycks/data.tar`；下载后记录 tar 文件 SHA256。
- 需要记录：commit hash 或下载日期、dataset version、科目分层策略、final 1000 sample ids hash、split hash。
- 注意：Edge 与 Cloud 必须使用相同 prompt、parser 和采样参数。

### CMMLU

- 目录：`data/datasets/cmmlu`
- 主来源：作者 GitHub 仓库 `haonan-li/CMMLU`，Hugging Face 数据页作为作者链接的分发入口。
- 下载方式：优先从作者 GitHub 的 `data` 目录或作者链接的 Hugging Face 数据页获取。
- 需要记录：commit hash 或下载日期、dataset version、final 1000 sample ids hash、synthetic 数据 hash。
- 注意：作者 GitHub README 写明 CC BY-NC-SA 4.0；Hugging Face 元数据当前显示 `cc-by-nc-4.0`，下载前以作者仓库 license 说明为准并记录复核结果。final evaluation 子集不得进入蒸馏；synthetic 需记录 prompt hash、generation seed、teacher model 和 sample hash。

### MVTec AD

- 目录：`data/datasets/mvtec_ad`
- 主来源：MVTec 官方 MVTec AD 页面。
- 下载方式：官方页面要求填写表单下载；不要使用未说明来源的镜像替代官方数据。
- 需要记录：下载日期、表单/协议确认、dataset version、train normal 样本数、official test 是否全量进入 Final Gate。
- 注意：官方页面标明 CC BY-NC-SA 4.0 且禁止商业用途。validation 使用 train 正常子集、NEU-DET val 缺陷样本和非 final 派生异常样本，不使用官方 test 做 validation。

### NEU-DET

- 目录：`data/datasets/neu_det`
- 主来源：东北大学 Song Kechen 教师主页的 NEU surface defect database 页面。
- 下载方式：优先访问官方教师主页；如果页面在当前网络环境中超时，再记录失败时间、HTTP 情况和替代镜像来源，不直接把镜像视为官方来源。
- 需要记录：下载日期、source URL、license/terms 复核结果、dataset version、每类 train/validation/test 数量、类别分层 split hash。
- 注意：计划中 final test 为 360，train 使用 70%。当前仅完成主来源定位，license/terms 需要下载前人工复核。

### CityFlow

- 目录：`data/datasets/cityflow`
- 主来源：AI City Challenge 2022 Track 1 CityFlowV2 数据页。
- 下载方式：官方页面提供 Track 1 data download，点击即接受 data license agreement；必须人工确认协议后下载。
- 需要记录：下载日期、data license agreement 确认、dataset version、camera/intersection 子集、vehicle id 数、relation edge/group/conflict group 统计。
- 注意：当前项目计划优先使用 CityFlow-Original；若实际使用 CityFlowV2，必须在 `dataset_manifest.json` 记录实际版本。若官方 test 标签不可用，按 scene/camera/time window 冻结划分并记录 split hash。

### UA-DETRAC

- 目录：`data/datasets/ua_detrac`
- 主来源：UA-DETRAC 官方项目入口 `detrac-db.rit.albany.edu`，当前重定向到 University at Albany CVML 页面；UBMDFL downloads 页面也保留 UA-DETRAC Project Page 入口。
- 下载方式：从官方项目页或 UBMDFL downloads 入口进入后下载；下载前确认研究用途、license/terms 和文件列表。
- 需要记录：下载日期、source URL、license/terms 复核结果、dataset version、视频/帧/标注统计、final sample ids hash。
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

## Source Links

[gsm8k-source]: https://github.com/openai/grade-school-math
[humaneval-source]: https://github.com/openai/human-eval
[mmlu-source]: https://github.com/hendrycks/test
[cmmlu-source]: https://github.com/haonan-li/CMMLU
[cmmlu-hf]: https://huggingface.co/datasets/haonan-li/cmmlu
[mvtec-source]: https://www.mvtec.com/research-teaching/datasets/mvtec-ad
[neu-source]: https://faculty.neu.edu.cn/songkc/en/zdylm/263265
[cityflow-source]: https://www.aicitychallenge.org/2022-data-and-evaluation/
[uadetrac-source]: https://detrac-db.rit.albany.edu/
[uadetrac-downloads]: https://ubmdfl.cse.buffalo.edu/index.php?page=downloads
