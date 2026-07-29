# Dataset Sources

核验日期：2026-07-29。

## 当前有效能力数据

| Dataset | Purpose | Source | Terms |
|---|---|---|---|
| GSM8K | Math训练、门禁和正式测试 | https://github.com/openai/grade-school-math | repository MIT |
| OpenCodeInstruct | Code训练和内部门禁 | https://huggingface.co/datasets/nvidia/OpenCodeInstruct | CC BY 4.0 |
| HumanEval | Code正式测试 | https://github.com/openai/human-eval | repository MIT |
| COIG-CQIA | 中文NLP训练 | https://huggingface.co/datasets/m-a-p/COIG-CQIA | dataset card license待补充 |
| CMMLU | NLP小门禁和正式测试 | https://github.com/haonan-li/CMMLU | upstream non-commercial terms |

OpenCodeInstruct只下载配置中固定的三个分散parquet分片。COIG-CQIA使用官方完整JSONL，但只筛选人工核验且属于五类目标领域的样本。

原始数据、筛选数据和逐题输出均不进入Git。仓库只提交重建脚本、配置、公开汇总审计与哈希。

下载和构建：

```bash
bash scripts/run_p0a5.sh data-download
bash scripts/run_p0a5.sh data-build
```

## 当前场景数据

| Dataset | Purpose | Source/Status |
|---|---|---|
| NEU-DET | 工业表面缺陷 | 东北大学来源；本地镜像结构已核验 |
| MVTec AD | 工业异常检测 | MVTec官方页；CC BY-NC-SA 4.0 |
| CityFlow | 交通多摄像头关联 | AI City官方数据协议 |
| UA-DETRAC | 交通检测补充 | 官方入口不可直接下载，本地镜像结构已核验 |

上述四个场景数据保留，不参与Math/Code/NLP能力蒸馏。

## 已退出主线

以下数据不再下载、不再训练、不再作为当前门禁：

- MMLU及其中文翻译；
- MBPP；
- APPS；
- CodeContests；
- 旧96题和170题组合门禁。

历史报告可以保留路线结论，但相关原始数据可以重新下载，因此本地旧副本已删除。
