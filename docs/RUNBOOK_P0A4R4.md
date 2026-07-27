# P0-A4R4 长代码蒸馏修复

P0-A4R4只在P0-A4R3两个预注册候选均完成任务级评测且Code门禁失败后启用。
它不读取旧42题、96题、170题或正式全测逐题输出。

本路线针对P0-A4R3暴露的两个结构问题：

- Code训练来源为MBPP 292、APPS 2500、CodeContests 1000，但未经损失均衡，
  APPS实际主导共享训练；
- CodeContests验证为长题，失败响应经常生成到1024 token上限，单题约32秒。

修复策略：

- 对CodeContests的多个执行通过参考解选择token最短者；
- 参考答案最多768 token，prompt与答案总长仍不超过1536 token；
- MBPP、APPS、CodeContests不复制样本，通过`training_weight`获得相等总损失质量；
- Code新验证集使用从未进入P0-A4R3训练或验证的APPS 128题和CodeContests
  128题；
- NLP和Math也建立新的train-only验证组；
- 推理侧Code最大生成长度预注册为768，不在看到验证结果后修改。

先确认上一轮已经结束：

```bash
bash scripts/run_p0a4r3.sh select || true
```

构建紧凑Code训练数据和新验证集：

```bash
bash scripts/run_p0a4r4.sh apps-fresh-validation
bash scripts/run_p0a4r4.sh code-fresh-validation
bash scripts/run_p0a4r4.sh code-compact-train
```

NLP新验证集需要已启动的4 GPU 14B Teacher端点：

```bash
bash scripts/run_p0a4r4.sh nlp-prepare
bash scripts/run_p0a4r4.sh nlp-generate
```

随后组装、预检并先运行v1新分母：

```bash
bash scripts/run_p0a4r4.sh assemble
bash scripts/run_p0a4r4.sh preflight
bash scripts/run_p0a4r4.sh evaluate-base
```

最多训练两个预注册共享候选：

```bash
bash scripts/run_p0a4r4.sh train 1
bash scripts/run_p0a4r4.sh merge 1
bash scripts/run_p0a4r4.sh evaluate 1
bash scripts/run_p0a4r4.sh select || true
```

仅当Candidate 1未通过时，才运行Candidate 2：

```bash
bash scripts/run_p0a4r4.sh train 2
bash scripts/run_p0a4r4.sh merge 2
bash scripts/run_p0a4r4.sh evaluate 2
bash scripts/run_p0a4r4.sh select
```

候选仍须满足Math相对v1不低于95%、Code/NLP均不回退且二者平均至少净增
1个百分点。未通过新门禁不得进入96/170题或正式全测。
