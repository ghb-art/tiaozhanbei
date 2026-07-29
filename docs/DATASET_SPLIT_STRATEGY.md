# Dataset Split Strategy

当前能力协议使用`seed=20260729`，场景数据继续使用原已冻结划分。真实数据和生成逐题记录不进Git。

## 能力数据

| Dataset | Train | Internal validation | Small gate | Formal test |
|---|---:|---:|---:|---:|
| GSM8K | official train 7,173 | official train 200 | official train 100 | official test 1,319 |
| OpenCodeInstruct | filtered train 20,000 | filtered train 1,000 | filtered train 100 | none |
| HumanEval | none | none | none | official 164 |
| COIG-CQIA | human-verified stratified 9,500 | stratified 1,000 | none | none |
| CMMLU | none | none | official dev 100 | official test 11,582 |

规则：

- GSM8K正式test不参与任何开发。
- OpenCodeInstruct三组按ID和语义去重，并与HumanEval去重。
- COIG-CQIA训练与内部验证互斥，并与CMMLU精确去重。
- CMMLU dev门禁覆盖全部67个学科。
- 唯一能力小门禁为300题，不再维护96题和170题。
- 正式测试只允许输出任务级汇总用于报告，不反馈训练。

生成命令：

```bash
bash scripts/run_p0a5.sh data-build
```

冻结清单：

```text
data/capability_v2/manifest.json
data/capability_v2/gate300.jsonl
reports/audit/gate_p0a5_data.json
reports/audit/gate_p0a5_protocol.json
```

## 场景数据

| Dataset | 用途 | 划分 |
|---|---|---|
| NEU-DET | 工业缺陷主场景 | 每类210/30/60 |
| MVTec AD | 工业异常补充 | official train用于开发，official test用于最终场景评测 |
| CityFlow | 交通多摄像头主场景 | official train/validation/test |
| UA-DETRAC | 交通检测补充 | 保留官方风格train/test |

场景数据不得与能力训练混合。Fast Path、弱网和冲突实验使用独立场景清单及哈希。
