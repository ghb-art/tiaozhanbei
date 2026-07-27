# P0-A4R 推理蒸馏与代码泛化修复手册

## 1. 目标与边界

P0-A4R不是第三个170题Student版本，而是在第二个Student尚未消耗170题名额前，对其任务
Adapter进行train-only修复。所有checkpoint选择只使用重新冻结的训练内验证集：

- 不读取96题逐题输出，只知道此前公开的三任务汇总；
- 不读取170题逐题输出；
- 不读取正式完整测试逐题输出或标签；
- NLP使用训练标签验证Teacher最终选项；
- Code使用可执行train-only测试选择checkpoint。

协议入口：

```bash
bash scripts/run_p0a4r.sh preflight
```

## 2. NLP：短理由与直接选项混合蒸馏

原流程要求Teacher只输出`A/B/C/D`，没有产生任何解释。新流程从512个train-only
CMMLU/MMLU辅助组中确定性留出64组，仅用于内部直接选项准确率；其余448组每题请求两个
不同提示的短理由：

```text
理由：……
FINAL: C
```

只有`FINAL`与训练标签一致、理由长度在12–320字符且不含“标准答案/ground truth”等泄漏
表达时才接受。每组再加入一条纯字母回放，因此预期监督比例约为`2份理由 : 1份直接答案`。

先在终端A启动已冻结的Teacher路由：

```bash
P0A4_TEACHER_VERSION=1 bash scripts/run_p0a4.sh teacher-serve
```

终端B生成数据：

```bash
bash scripts/run_p0a4r.sh nlp-generate
```

该命令支持断点续跑。dry-run trace不会被正式`--resume`误用。

## 3. Code：独立任务、执行验证和canonical回退

当前可立即恢复的数据不是`233×4`，而是P0-A4 train-only池中的292个独立MBPP canonical
任务；每个canonical答案都会重新执行完整存储测试。内部checkpoint选择使用从未进入当前
Student训练、96题或170题的MBPP `dev_gate` 42组。

```bash
bash scripts/run_p0a4r.sh code-build
```

若只有292组，审计会显示：

```text
status=passed
training_scope=pilot_only
promotion_eligible=false
```

这允许验证“低rank、低学习率、执行选优”是否有效，但不得进入96/170。要进入promotion，
如果存在外部备份，可以恢复既有
`data/distill/code_v30_validated_train.jsonl`，恢复后必须匹配冻结SHA256
`757ef8babd598af12f58a75c732020e5f8ef2b623c6ca07fb3114b0bf32f6fad`。该文件被
`.gitignore`排除，不能从Git/GitHub恢复；没有备份时必须重新构建新的train-only可执行
数据，并通过
`P0A4R_EXTRA_CODE_SOURCES`提供同一标准JSONL结构的train-only数据：

```bash
P0A4R_EXTRA_CODE_SOURCES=data/distill/my_verified_apps.jsonl,data/distill/my_verified_code_contests.jsonl \
  bash scripts/run_p0a4r.sh code-build
```

当前项目提供了更轻量的APPS官方train-only重建入口，不需要下载约7.5GB的
CodeContests：

```bash
bash scripts/run_p0a4r.sh code-source-rebuild
bash scripts/run_p0a4r.sh code-source-status
bash scripts/run_p0a4r.sh code-build
```

重建器验证官方归档SHA256，只解压`APPS/train`，静态拒绝文件、网络、进程等危险操作，
并在受CPU、内存、文件大小限制的隔离子进程中执行全部存储测试。默认输出1500个独立、
执行通过且不与内部开发题冲突的任务到
`data/distill/p0a4r_apps_verified_train.jsonl`。HumanEval仅用于无标签、无测试的prompt
去污染，不读取答案、测试或历史逐题结果。

2026-07-23实际重建结果：

- 官方归档SHA256：`6ef8e98ecca10b0159df0da4b524ecc1ca782a3b9473c57fc547ebccbbc2d0ca`；
- APPS train目录5000个，其中4805个具备完整构建文件；
- 2712个函数调用候选，2645个canonical执行通过；
- 冻结1500个APPS独立训练组，输出SHA256：
  `daae8d70f465473908cac64de8048e9bbbaad76ccbf5c2813665d05696432c1f`；
- 与292个MBPP组统一复验后得到1792个训练组和42个内部验证组，组重叠0、正式引用0；
- 最终Code训练集SHA256：
  `35d19790187e00b43b46513dd641f9bc0b8d1a8bf94d25dac91be4888202db8d`；
- `promotion_eligible=true`。

每行至少包含：

```json
{
  "dataset_key": "humaneval",
  "sample_id": "train-only-id",
  "validation_group_id": "unique-problem-id",
  "messages": [{"role": "user", "content": "function prompt"}],
  "answer": "verified Python completion",
  "code_eval": {
    "kind": "mbpp_assert_tests_v1 | apps_call_tests_v1 | code_contests_io_tests_v1",
    "entry_point": "function_name",
    "prompt_source": "def function_name(...): ...",
    "setup_code": "",
    "tests": ["executable stored tests"]
  },
  "used_for_training": true
}
```

构建器按`validation_group_id`只保留一行，重新执行canonical答案，拒绝正式测试、96和170
身份，并要求promotion训练至少1000个独立组。APPS与CodeContests只能使用官方train split。

## 4. 训练和内部checkpoint选择

Code production训练：

```bash
P0A4R_GPUS=0,1,2,3 bash scripts/run_p0a4r.sh train-code
```

只有292组时可以运行方法学试验：

```bash
P0A4R_GPUS=0,1,2,3 bash scripts/run_p0a4r.sh train-code-pilot
P0A4R_EVAL_GPU=0 bash scripts/run_p0a4r.sh eval-code-pilot
bash scripts/run_p0a4r.sh select-code-pilot
```

pilot使用独立的`code-pilot`模型、评测和选择目录；结果无论多高都不能被
`prepare-router`消费，也不能进入96题。NLP训练：

```bash
P0A4R_GPUS=0,1,2,3 bash scripts/run_p0a4r.sh train-nlp
```

两个任务都固定使用：

- LoRA rank 8、alpha 16、dropout 0.05；
- 学习率`3e-5`，两轮；
- 四卡DDP，单卡batch 1，梯度累积8，有效batch 32；
- 每轮保留checkpoint，不再由token级`eval_loss`直接发布。

逐checkpoint在单卡上运行train-only内部生成评测：

```bash
P0A4R_EVAL_GPU=0 bash scripts/run_p0a4r.sh eval-code
P0A4R_EVAL_GPU=0 bash scripts/run_p0a4r.sh eval-nlp
bash scripts/run_p0a4r.sh select-code
bash scripts/run_p0a4r.sh select-nlp
```

候选必须比共享v2基座净增至少1题，且回退不超过2题。Code用实际执行通过率，NLP用直接选项
准确率；未达标时不会发布Adapter。

## 5. Top-1路由和最后一次96题

只有Code数据审计`promotion_eligible=true`且两个内部选择门都通过，才准备路由：

```bash
bash scripts/run_p0a4r.sh prepare-router
bash scripts/run_p0a4r.sh edge-start
bash scripts/run_p0a4r.sh smoke96
bash scripts/run_p0a4r.sh edge-stop
```

路由继续使用现有Math v2 Adapter，替换Code和NLP Adapter。`smoke96`使用独立
`adapter_remediation`产物名，并拒绝第二次运行，避免围绕同一96题汇总持续调参。只有三项
及宏保持率均达到预注册安全线后，才另行授权最后一次170题。

`prepare-router`会再次检查Code数据的`promotion_eligible=true`，即使pilot内部评测通过，
也无法绕过该门禁；同时还会核对Code/NLP训练集、内部验证集和checkpoint选择报告的哈希
血缘，禁止补充新数据后误用旧Adapter。
