# 60% 内存看门狗

`scripts/run_with_memory_guard.sh`用于保护训练、评测或模型服务任务。默认每
2秒检查一次主机 RAM；主机 RAM 连续两次达到或超过60% 时，先向被保护
任务的独立进程组发送 `SIGTERM`，10秒后仍未退出则发送`SIGKILL`。GPU
显存不参与采样、判断或任务终止。

看门狗只会终止由它启动的任务及其子进程，不会扫描或终止服务器上的其他
用户作业。任务必须通过该脚本启动才能获得主动保护。

```bash
bash scripts/run_with_memory_guard.sh \
  bash scripts/run_p0a4r2.sh train-code
```

查看当前主机 RAM 使用率，不启动任务：

```bash
.venv/bin/python scripts/memory_watchdog.py snapshot
```

每次运行的状态文件写入`reports/runtime/memory_watchdog_*.json`，事件日志
追加到`logs/memory_watchdog.log`。因内存超限终止时，脚本退出码为75。

如需调整采样频率或连续次数，可设置环境变量，但正式运行保持60%阈值：

```bash
MEMORY_GUARD_INTERVAL_SECONDS=1 \
MEMORY_GUARD_CONSECUTIVE_SAMPLES=3 \
bash scripts/run_with_memory_guard.sh COMMAND
```
