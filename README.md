# mini_mega

fp8xfp8 MoE Mega Kernel 的**最小可复现子集**，从
`megakernel/zzzz/projs_deepseek/projs/01.fp8xfp8_moe_kernel` 抽取本次
R=32 TPR=16/64/256 性能测试（test/70）所涉及的全部文件。

## 内容

```
mini_mega/
├── src/                       # 10 个源文件（bench.cu + 9 个 .cuh）
├── include/deep_gemm/         # 最小外部头（mega_moe.cuh + common/{math,exception}.cuh）
├── scripts/build.sh           # cucc 编译；优先用本工程 include/，缺失回退 MXMOEMega
├── docs/02_测试流程与性能标准.md
├── perf.md                    # R=32 性能基准口径与基线
├── test/70.r32_tpr_perf/      # 测试脚本/日志/结果
│   ├── run.sh  results.md  test_info.md
│   └── r{1,2}.log               # 2 次运行原始日志
└── README.md
```

## 构建

```bash
setarch $(uname -m) -R bash scripts/build.sh   # 确定性构建 -> ./bench
```

依赖：MACA（`MACA_PATH=/opt/maca`）、`cucc`、`mcr` 运行时、系统头 `cute`
（位于 `/opt/maca/include`）。无需 torch / 容器。

## 测试（R=32 TPR=16/64/256）

```bash
bash test/70.r32_tpr_perf/run.sh
```

口径（对齐 `mega/src/bench/bench_mega_fp8.cu`）：单进程 `run_aligned`，均衡路由，
graph 链 `nodes=20`、`replays=2`（取 min）、默认不做正确性校验：

```bash
MACA_VISIBLE_DEVICES=0,...,31 ./bench --ranks 32 --tprs 16,64,256 --nodes 20 --replays 2
```

`./bench` 不带参数 = 默认全档 R=32 TPR={1,4,16,32,33,64,128,256}（单次约 2.9s，
满足 10s 约束）。`--aligned 0` 可退回旧的单档/扇形口径。

## 基准（本机 202，C600UL 32 卡）

| TPR | 16 | 64 | 256 |
|---:|---:|---:|---:|
| full (µs/kernel) | 343.5 | 651.0 | 1918.3 |

详见 `test/70.r32_tpr_perf/results.md`。
