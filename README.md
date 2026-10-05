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
│   ├── t{16,64,256}_r{1,2}.log   # 重编译后运行
│   └── run1/t*.log               # 重编译前运行
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

口径：`--ranks 32 --nodes 20 --graph 0 --perf-only 1 --compute 1 --perf-pattern 0
--tokens T --max-tokens 512 --win-cap 128`；单档单次 `timeout 10`。

## 基准（本机 202，C600UL 32 卡）

| TPR | full (µs/kernel) |
|---:|---:|
| 16 | 343.8 |
| 64 | 655.3 |
| 256 | 1913.6 |

详见 `test/70.r32_tpr_perf/results.md`。
