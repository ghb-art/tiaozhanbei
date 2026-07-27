# P0-A4R2 v1 Code温和修复手册

## 1. 固定路线

- 基座：`models/checkpoints/p0a4/student-shared-merged`（v1）。
- 量化模型：v1 Q4_K_M，KV固定Q8。
- Math与NLP：共享v1，不加载任务Adapter。
- Code：只有通过train-only内部执行选择的Adapter才允许加载。
- v2 NLP Adapter：按哈希冻结归档，不能跨基座用于v1。

冻结和预检：

```bash
bash scripts/run_p0a4r2.sh freeze-nlp
bash scripts/run_p0a4r2.sh preflight
bash scripts/run_p0a4r2.sh train-code-dry
```

## 2. 数据平衡与训练

训练集保持1792个独立任务，不生成重复训练行：

- APPS official train：1500组；
- MBPP train：292组；
- 内部验证：独立MBPP dev_gate 42组。

训练器按来源计算逐样本损失权重，使APPS和MBPP的总损失质量相同。当前权重为：

- APPS：`0.5973333333`，总质量约896；
- MBPP：`3.0684931507`，总质量896。

正式训练：

```bash
P0A4R2_GPUS=0,1,2,3 bash scripts/run_p0a4r2.sh train-code
```

固定参数为Rank 4、alpha 4、dropout 0.05、学习率`1e-5`、1轮、单卡batch 1、
梯度累积8。训练输出不可覆盖；需要新试验时必须使用新的候选目录和审计名。

## 3. 仅用内部验证选择

```bash
P0A4R2_EVAL_GPU=0 bash scripts/run_p0a4r2.sh eval-code
bash scripts/run_p0a4r2.sh select-code
```

选择门槛：

- 相对v1基线净增至少1题；
- 原本正确题回退不超过2题；
- 生成错误为0；
- 只允许读取42题train-only内部执行结果。

首次Rank-4候选的结果为：

- v1基线：24/42；
- checkpoint-56：24/42；
- 新增0、回退0、净增0；
- 选择失败，没有创建`models/checkpoints/p0a4r2-v1/code-selected`。

因此该Adapter不能转换为GGUF、不能进入96题，也不能替代纯v1部署路线。
