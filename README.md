# mini_mega

fp8xfp8 MoE Mega Kernel 的**最小可复现子集**，从
`megakernel/zzzz/projs_deepseek/projs/01.fp8xfp8_moe_kernel` 抽取本次
R=32 TPR=16/64/256 性能测试（test/70）所涉及的全部文件。

## 内容

```
mini_mega/
├── src/
│   ├── megakernel.h            # host C++ 接口（公共头，bench 唯一依赖）
│   ├── megakernel.cu           # 接口实现 + kernel 发射 -> megakernel.so
│   ├── bench.cu                # benchmark：只调接口，不直接发射 kernel
│   └── *.cuh                   # kernel 实现（kernel/prologue/compute/combine/flags/...）
├── include/deep_gemm/          # 最小外部头（mega_moe.cuh + common/{math,exception}.cuh）
├── scripts/build.sh            # 先编 megakernel.so，再链接 bench
├── docs/02_测试流程与性能标准.md
├── perf.md                     # R=32 性能基准口径与基线
├── test/70.r32_tpr_perf/       # 均衡 TPR 基准
│   ├── run.sh  results.md  test_info.md
│   └── r{1,2}.log                # 2 次运行原始日志
├── test/71.r32_tk_imbalance/   # 真实 TK_INFO 不均衡路由基准
│   ├── run.sh  results.md  run.log
└── README.md
```

## 分层

- **接口层**：`src/megakernel.h`（声明 `fp8mega::MegaLaunchArgs` 与
  `launch_prelude/compute/reduce/full`、`launch_empty`、`fill_weights_lane`）。
- **实现层**：`src/megakernel.cu` 是**唯一**发射 `mega_kernel` /
  `mega_reduce_kernel` / `mega_empty_kernel` / `fill_w_lane_kernel` 的地方，
  编译为 `megakernel.so`。
- **调用层**：`src/bench.cu` 只 `#include "megakernel.h"` 并链接 `megakernel.so`，
  不包含 `kernel.cuh`、不出现任何 `<<<>>>`。

## 构建

```bash
setarch $(uname -m) -R bash scripts/build.sh   # -> ./megakernel.so + ./bench
```

依赖：MACA（`MACA_PATH=/opt/maca`）、`cucc`、`mcr` 运行时、系统头 `cute`
（位于 `/opt/maca/include`）。无需 torch / 容器。运行 `./bench` 时经 rpath
`$ORIGIN` 自动加载同目录的 `./megakernel.so`。

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

## 不均衡测例（真实 TK_INFO）

读取内核抽样日志 `0928_tk_info.log`，复现某次 launch 的每专家收行分布并建图测量：

```bash
bash test/71.r32_tk_imbalance/run.sh
./bench --tk-info test/70.r32_tpr_perf/0928_tk_info.log --tk-launch 853000
```

日志行 `[TK_INFO] rank=r launch=n : c0..c7`；`ci` = rank r 物理专家 i 收到的 pair
数。`--tk-launch N` 指定 launch（缺省自动选「单专家最大收行」最大的全 rank
launch）。实测单次全链：均衡 354.6µs → 不均衡 1.81/3.46/4.72ms（max 176/362/512），
详见 `test/71.r32_tk_imbalance/results.md`。

## 基准（本机 202，C600UL 32 卡）

| TPR | 16 | 64 | 256 |
|---:|---:|---:|---:|
| full (µs/kernel) | 343.5 | 651.0 | 1918.3 |

详见 `test/70.r32_tpr_perf/results.md`。

## flag 结构体改造（local_flags / ranks_flags + current_time）

按参考实现 `mega/src/kernel/mega_fp8.cuh` 重构同步标志（2026-10-05，未提交 git）：

- 新增 `src/flags_defs.cuh`：
  - `struct alignas(256) local_flags { int64_t current_time[kGrid][64];
    uint32_t bar[1032]; unsigned last_ready[kGrid]; }`（普通显存，本 rank 私有）；
  - `struct alignas(256) ranks_flags { uint32_t xbar_barrier[kGrid][kGrid]; }`
    （FG 显存，跨 rank peer 可写）。
- `MegaArgs` / `MegaLaunchArgs` 的 `bar_flags` 改 `local_flags* lf`，
  `epoch_ctr/last_release` 由每 block 单调 launch 计数 `lf->current_time[b][*]` 取代，
  `last_ready` 并入 `lf->last_ready`；xbar 跨 rank 旗从 sym 迁到 `rf->xbar_barrier`。
- 行为不变：`bar[]` 索引逐一对齐原 `kBar*Idx`；可见性顺序（payload→fence→ready）
  与自旋/超时语义保持。

正确性（旧口径 R=2/8/32，`--perf-only 0 --compute 1`）全部 `[总体] PASS`；
性能（test/70，graph，4 次采样均值）：TPR=16/64/256 = 350.3/644.9/1906.6 µs/kernel，
相对改前基线 353/651/1915 均略快（全在 ±3% 内，见 `perf.md` v2）。
