# Project Status

更新时间：2026-07-23

## 阶段结论

项目仍位于 P0，当前没有候选同时通过 G1 能力保持率和 G3 峰值内存。不得进入正式系统主线或声称边缘模型已冻结。

| 角色 | 模型 | 当前结论 |
|---|---|---|
| Baseline-14B-AWQ | Qwen2.5-14B-Instruct-AWQ | 不可调优，只作为正式能力分母；单端点、GPU 0–3、tensor parallel 4 |
| Distill-Teacher-14B | Qwen2.5-14B-Instruct BF16 + LoRA任务路由 | v1已完成4卡ZeRO-3训练；验证后冻结为Math/Code走BF16基座、NLP走v1 LoRA的Top-1路由 |
| Edge-Student v1 | Qwen3-1.7B 蒸馏版 Q4_K_M + Q8 KV | 96题和5+20内存预检通过；170题Code保持率75%而失败；提前执行的一次官方全测也失败并已封存 |
| Edge-Student v2 | Qwen3-1.7B共享蒸馏 + Top-1任务Adapter | 共享模型和三个Adapter均已训练、合并/转换并完成96题；共享与Adapter路线都因Code保持率70.83%失败，未消耗第二次170题机会 |
| P0-A4R修复 | v2共享基座 + 新Code/NLP Adapter | NLP修复后的96题保持率达到90%，已按模型、Adapter和报告哈希冻结；该NLP Adapter仅兼容v2，不挂载到v1。修复版Code保持率只有66.67%，整条v2 Adapter路线终止 |
| P0-A4R2回退 | v1 Q4_K_M共享模型 + 可选Code Adapter | 已回到表现更好的v1；Math/NLP走共享v1。使用1792个独立任务完成Rank 4、alpha 4、`1e-5`、1轮、4卡DDP Code训练，MBPP/APPS不复制样本而按总损失各占50%平衡；内部执行结果为基线24/42、Adapter 24/42，净增0且回退0，故Adapter未发布，当前可部署路线仍为纯v1 |
| P0-A3 主候选 | Qwen3-1.7B Q3_K_M + Q8 KV | Teacher与HF 170条均完成；HF保持率Math/Code/NLP/Macro=`87.50/83.87/86.96/86.11%`。原Q3+Q4 KV完整结果仅`43.75/41.94/45.65/43.78%`，F16+Q4 KV也只有`53.13/58.06/63.04/58.08%`；token完全一致但首token logits偏移。改为F16 KV后logits与HF恢复一致，Q3+Q8 KV 24题为`3/8、7/8、3/8`，接近HF `3/8、7/8、4/8`。主候选重新开放，待Q8 KV完整170条。 |
| P0-A3 备用 | Qwen2.5-1.5B-Instruct Q3_K_M | 原Q4 KV拒绝已判定为运行参数污染，不再解锁备用；只有Qwen3 Q8 KV完整能力或内存Gate明确失败后才启动。 |
| 已拒绝 | DeepSeek-R1-Distill-Qwen-1.5B | BF16 Dev：Math 84.38%、Code 23.81%、NLP 17.19%；P0-A2 关闭，不再训练 |

DeepSeek Q2 的 716.71MB 文件和 920.23MB 运行峰值只证明内存安全，不能抵消能力失败。完整执行报告固定为 `gate_p0a2_deepseek_upper_bound.json`，路线结论固定为 `gate_p0a2_deepseek_rejection.json`；后者明确为 `failed/close`，避免把前者的执行 `passed` 误读为能力通过。

## P0-A4 已落地代码

- 独立冻结官方完整清单：GSM8K 1319、HumanEval 164、CMMLU 11582，总计13065；不覆盖历史500/164/1000清单。
- 从训练池按组留出Teacher验证96和Student烟测96；它们与训练集、170题选择集之间的组重叠均为0。
- 防泄漏训练器拒绝`reports/sealed`、`data/eval`、官方完整split及任何正式测试身份标记。
- 14B BF16 LoRA使用4卡ZeRO-3；Student支持共享蒸馏LoRA、合并和可选的三个Top-1任务Adapter。
- 量化固定为训练集imatrix、Q4_K_M权重和Q8 KV；Q3_K_M权重因代码保持率失败仅保留对照，Q4 KV Cache继续禁用。
- 96题要求每项及宏平均≥75%；170题要求每项及宏平均≥80%，只输出任务级汇总且最多两个Student版本。
- 正式基线和Student采用四个确定性分片，合并时检查重复、遗漏和额外ID；逐题trace封存，Student正式尝试上限为1。
- BF16 Teacher已下载并完成v1训练。单一v1 LoRA在独立96题上为Math/Code/NLP=`32/23/31`，代码项相对AWQ基线`32/26/31`回退；第2轮代码进一步降至`22/32`，故不采用全任务LoRA。
- 冻结的Teacher v1任务路由在同一96题上达到`32/26/31`、宏准确率`92.708%`、生成错误0，`gate_p0a4_teacher_selection.json`已通过。该结论只代表独立Teacher验证门，不代表Student或竞赛最终指标通过。
- Teacher正确性过滤蒸馏已完成。修复了旧训练池将16个CMMLU问题复制4096次的问题；新NLP池包含512个唯一问题且与96/170/正式测试无重叠。最终有效蒸馏数据为Math/Code/NLP=`1952/233/440`，逐任务接受率均超过75%，Student训练预检平衡为每类466条。
- 共享Student v1已完成4卡DDP LoRA训练，63/63步通过；最佳验证损失为checkpoint-21的`0.580263`，后两轮分别升至`0.650273/0.668305`，因此已发布并合并checkpoint-21。量化imatrix固定从Math/Code/NLP各取128条训练数据，正式测试引用为0。
- Student v1 HF诊断96题保持率Math/Code/NLP/Macro=`75.00/79.17/80.00/78.06%`；修正审计口径后的Q3仍仅为`71.88/50.00/85.00/68.96%`，判定能力失败。Q4为`87.50/75.00/95.00/85.83%`并通过96题门；关闭llama.cpp主机提示缓存后的5+20内存峰值为`1211.20MB`，通过1400MB预检。
- 正式AWQ完整分母已完成并密封：GSM8K=`91.736%`、HumanEval=`64.024%`、CMMLU=`75.928%`。
- Student v1的170题保持率为Math/Code/NLP/Macro=`90.625/75.000/80.435/82.020%`，生成错误0；因Code低于80%失败。一次官方全测已按显式绕过授权提前完成并封存，保持率为`82.149/68.571/70.650/73.790%`，不得用于v2训练或选择。
- v2预检已通过：共享训练目标数为Math/Code/NLP=`466/932/880`；Adapter rank分别为`4/16/12`。共享与Adapter路线均需重新跑96题，只有各项及宏都达到85%才允许消耗最后一个170题候选名额。
- v2实际96题结果已经完成：共享路线Math/Code/NLP/Macro=`78.125/70.833/85.000/77.986%`，原Top-1 Adapter路线=`75.000/70.833/85.000/76.944%`，两者均失败且没有消耗第二次170题机会。P0-A4R禁止读取逐题错误，仅使用train-only内部验证。
- P0-A4R APPS重建已完成：官方归档SHA256=`6ef8e98e...2d0ca`，仅解压train split；2712个函数调用候选中2645个canonical执行通过，去污染和Qwen3 1536-token过滤后冻结1500组，输出SHA256=`daae8d70...32c1f`。
- P0-A4R Code数据门已实测：APPS 1500组 + MBPP 292组，共1792个独立canonical训练组；42个独立执行验证组；组重叠0、正式引用0；`promotion_eligible=true`，训练集SHA256=`35d19790...02db8d`。NLP短理由数据、训练、内部选择和量化96题验证均已完成。
- P0-A4R NLP训练与内部选择已完成，量化v2修复路线96题NLP保持率从85%升至90%；冻结清单位于`models/adapters/p0a4r/frozen_nlp_manifest.json`。由于其训练基座是`student-shared-v2-merged`，不得应用到v1。
- P0-A4R2已使用`student-shared-merged`（v1）完成温和Code训练。原始数据仍为APPS 1500 + MBPP 292个独立组，训练器采用损失权重`0.597333`与`3.068493`，两类总损失质量均为896，`row_duplication=false`。唯一checkpoint-56在42题train-only执行集上与v1基线均为24题正确，未满足净增1题的发布门槛；`models/checkpoints/p0a4r2-v1/code-selected`未创建。

## P0-A3 历史证据

- `configs/p0a3_reselection.json` 固定 Teacher、170 条 Dev、候选顺序和门槛。
- `configs/g0_capmem_candidates.json` 将 Qwen3 Q3 设为主候选、Qwen2.5-1.5B Q3 设为备用，DeepSeek 仅保留拒绝证据。
- `evaluate_edge_candidate_dev.py` 提供 HF/llama.cpp HTTP 同口径评测，并支持 Qwen3 `enable_thinking=false`。
- HTTP 评测会从 `/v1/models` 发现实际服务模型 ID；HF 评测会自动选择空闲显存最多的 GPU，已分别通过 3 条 Teacher/Qwen3 实机烟测。
- `summarize_edge_candidate_dev.py` 强制 Teacher/Candidate sample ID 完全一致，并对 Math、Code、NLP 分别执行 80% 保持率硬门。
- `run_p0a3.sh` 串联 Teacher 分母、HF 上限、importance-aware Q3、量化后同集复验、1400MB Dev 内存余量和一次正式 G0。
- `setup_llama_cpp.sh` 固定已审计的 llama.cpp 提交并应用 UTF-8→PEG 兼容补丁；补丁只将非法字节显式替换为 `�`，不修改模型logits、任务答案或评分器。
- `run_p0a2.sh` 只返回关闭提示，不能再启动 DeepSeek 训练。

## 冻结 Dev 口径

170 条 selection-only 数据包括 CMMLU official dev 64、GSM8K train 内部留出 64、MBPP dev_select 42。训练/验证 group overlap、正式集引用和 HumanEval prompt overlap 均为 0。该集合可以做候选选择，但不能根据错题生成监督。

## 下一退出条件

1. 保留纯v1作为当前回退服务，不加载未晋级的P0-A4R2 Code Adapter；
2. 不覆盖NLP冻结清单、P0-A4R2训练报告或内部选择失败报告；
3. 如继续Code修复，必须先注册新的train-only候选协议和独立验证证据，不能反复围绕已消费的96题调参；
4. 只有新候选先通过内部执行选择、量化96题三项门和5+20内存≤1400MB，才可消耗第二个也是最后一个170题候选名额；
5. 170题要求三任务及宏均≥80%；既有官方完整测试证据不得被覆盖或反馈训练。

CPU llama.cpp 秒级短输出不能满足 0.2s 端到端指标。模型通过 G0 后仍须依赖加速设备和结构化 fast path 完成系统时延目标。
