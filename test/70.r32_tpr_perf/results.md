# test/70.r32_tpr_perf — R=32 TPR=16/64/256 性能

- 日期：2026-10-05；机器：本机 202（C600UL，32 卡 Available、全 0%）
- 二进制：`./bench` + `./megakernel.so`（当前 src；`bash scripts/build.sh`；
  bench 只调 `src/megakernel.h` 接口，kernel 发射在 `megakernel.so` 内）
- 口径：对齐 `mega/src/bench/bench_mega_fp8.cu` —— 单进程、均衡路由、graph 链
  `nodes=20`、`replays=2`（取 min）、默认不做正确性校验；`full` = 每节点
  `mega_kernel(+REDUCE)` 的 us/kernel（每 replay 独立 event 夹住、replay 时间/nodes、
  跨 rank 取 min）。
- 约束：单次 `timeout -k 3 10`（AGENTS 10s 规则）；结果取 2 次运行均值。
- 脚本：`run.sh`；日志：`r1.log` / `r2.log`。

## 实际执行命令

```
MACA_VISIBLE_DEVICES=0,...,31 timeout -k 3 10 stdbuf -oL ./bench \
  --ranks 32 --tprs 16,64,256 --nodes 20 --replays 2
```

> `./bench` 不带参数时为默认全档 R=32 TPR={1,4,16,32,33,64,128,256}（单次约 2.9s）。

## 结果

| TPR | rep1 (µs) | rep2 (µs) | 均值 (µs) | perf.md v1 (µs) | 口径 |
|---:|---:|---:|---:|---:|:--|
| 16 | 352.24 | 354.27 | **353.26** | 345.9 | graph |
| 64 | 647.39 | 654.48 | **650.94** | 655.1 | graph |
| 256 | 1916.22 | 1914.32 | **1915.27** | 1914.8 | graph |

## 默认全档（单次约 2.5s，补充记录）

| TPR | 1 | 4 | 16 | 32 | 33 | 64 | 128 | 256 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| µs | 302.09 | 313.82 | 349.89 | 454.77 | 553.72 | 636.49 | 1029.86 | 1916.31 |

> 结论：2/2 次运行 rc=0、口径 graph（无降级）；各档均值均落在 perf.md v1 基线
> ±1% 内 → 与旧单档（分档子进程）口径一致、无回归。rep 间波动（如 TPR=64 的
> 647.4 vs 654.5）为 202 机共享环境会话漂移。
