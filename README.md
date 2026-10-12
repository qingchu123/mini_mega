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
├── scripts/build.sh            # 产物输出到 test/（so / bench / mybench / check / whl）
├── scripts/run.sh              # 统一启动（mpirun，每进程一张卡；默认持 GPU 锁）
├── test/mybench.cu             # MPI 性能基准（-s 1 均衡 / -s 2 TK 实录）
├── test/check.cu               # MPI 正确性测试（--smoke / --all）
├── test/mega_common.cuh        # 两者共用的设备资源 / 布局 / 路由构造
├── docs/02_测试流程与性能标准.md
├── perf.md                     # R=32 性能基准口径与基线
├── test/70.r32_tpr_perf/       # 均衡 TPR 基准
│   ├── run.sh  results.md  test_info.md
│   └── r{1,2}.log                # 2 次运行原始日志
├── test/71.r32_tk_imbalance/   # 真实 TK_INFO 不均衡路由基准
│   ├── run.sh  results.md  run.log
└── README.md
```

## 关键编译期参数（`src/common.cuh`）

| 参数 | 值 | 说明 |
|---|---|---|
| `KNSLICE_OVERRIDE`（`kNSlice`） | 4 | 每窗口任务切片数；NS=4 时 L1 每片 4 组、L2 每片 6 组。NS=8 已验证可用（`g_l2=3` 需走 `compute_big.cuh` 的 `[R35]` 单批 3 组分支）；NS=16 暂不可用（L2 N 组几何不整除，待设计） |
| `kRing` | 8 | 同专家并发窗上限 |
| `kEPR` / `kTopk` / `kHidden` / `kIntermediate` | 8 / 8 / 6144 / 2048 | 场景维度 |
| `kGrid` | 32 | grid 块数（= SM 数） |

切片/L1 进度旗区容量由 `include/deep_gemm/layout/mega_moe.cuh` 定义
（`kBarMaxNSlice=16`，两旗区各 1024 项，承载 EPR×ring×NSLICE≤1024）。
NS=8 下 `r19_small_task` 的 L2 `g_l2=3` 走单批 3 组路径（`[R35]`，见
`compute_big.cuh`），不能走 `[O3]` 的 3+3 两批。

## 分层

- **接口层**：`src/megakernel.h`（声明 `fp8mega::MegaLaunchArgs` 与
  `launch_prelude/compute/reduce/full`、`launch_empty`、`fill_weights_lane`）。
- **实现层**：`src/megakernel.cu` 是**唯一**发射 `mega_kernel` /
  `mega_reduce_kernel` / `mega_empty_kernel` / `fill_w_lane_kernel` 的地方，
  编译为 `megakernel.so`。
- **调用层**：`src/bench.cu` 只 `#include "megakernel.h"` 并链接 `megakernel.so`，
  不包含 `kernel.cuh`、不出现任何 `<<<>>>`。

## 构建

产物统一输出到 `test/`（二进制与 `megakernel.so` 同目录，靠 `rpath=$ORIGIN` 定位）。

```bash
scripts/build.sh so        # test/megakernel.so     （接口层 + kernel）
scripts/build.sh mybench   # test/mybench           （每进程一张卡的 MPI 性能基准）
scripts/build.sh check     # test/check             （MPI 正确性测试）
scripts/build.sh bench     # test/bench             （单进程多卡 benchmark，旧口径）
scripts/build.sh whl       # torch/dist/*.whl       （需 python+torch，PYTHON= 可指定）
scripts/build.sh all       # = so bench mybench check（不含 whl）
```

不带目标时等价 `all`；`mybench`/`check`/`bench`/`whl` 会自动先编 `so`。
`whl` 不进 `all`：它依赖 torch 环境，本机未装 torch 时会必然失败。

构建为**增量**：`so` 依据各 TU 的 `-MMD` 依赖按 mtime 跳过未变动对象
（改动只影响部分 TU 时仅重编该部分），`mybench`/`check` 仅在自身源码或
`megakernel.so` 变化时重链。无改动重跑 `all` <0.1s；`KERNEL_CFLAGS` 或
包含路径变化时自动全量重编（指纹存于 `build/obj_so/.buildflags`）。

依赖：MACA（`MACA_PATH=/opt/maca`）、`cucc`、`mcr` 运行时、系统头 `cute`
（位于 `/opt/maca/include`）。`so`/`mybench`/`check`/`bench` 无需 torch / 容器。

> 运行前必须 `export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/ompi/lib:$LD_LIBRARY_PATH`：
> 系统 `ld.so.cache` 中含 `/opt/maca-3.8.4/lib`，否则会与 `/opt/maca` 混版，
> 报 `libmcpti.so: undefined symbol: TRACER_RECORD_NUM`。`scripts/build.sh`
> 已内置该导出；`scripts/run.sh` 亦已内置。

## 运行

`scripts/run.sh [程序] [-R N] [参数...]`，程序为 `mybench`（默认）| `check` | `bench`。
内部用 `mpirun -np R`（每进程一张卡），默认持 `.agents/work/lock/lock_gpu` 避免多 Agent 争抢。

```bash
# 性能（mybench）
scripts/run.sh mybench -R 32 -s 1 -tprs 1,2,4,6,8,12,18,24,30,36,42,48,54,60,66,72,78,84,90,128,256   # 均衡（默认场景）
scripts/run.sh mybench -R 32 -s 2 -top 50                              # TK 实录前 50 个
scripts/run.sh mybench -R 32 -s 2 -cases 804900,807500                 # 指定 TK launch

# 正确性（check）
scripts/run.sh check --smoke     # 冒烟，<10s
scripts/run.sh check --all       # 全量（均衡 TPR 表 + 全部 TK launch），不限时
```

实测（R=32，2026-10-08）：

| 命令 | wall |
|---|---|
| `build.sh so` / `mybench` / `check` / `all` | 35s / 38s / 39s / 48s（后三者含重编 so） |
| `run.sh mybench -R 32 -s 1 -tprs 1..512` | 7.2s |
| `run.sh mybench -R 32 -s 2 -cases 804900,807500` | 6.7s |
| `run.sh mybench -R 32 -s 2 -top 50` | 13.1s |
| `run.sh check --smoke` | 8.0 ~ 9.0s（3 次） |
| `run.sh check --all` | 约 4 分钟（762 用例） |

`check --all` 结果见 `test/77.correctness/测试流程.md`（L0/REDUCE/graph 全 PASS；
36 例 TK 用例 L1 未解决）。

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
launch）。实测单次全链：均衡 334.9µs → 不均衡 0.656/0.670/0.682ms（max 176/362/512，
v9 前序复用后；改前为 1.78/3.46/4.71ms），详见 `test/71.r32_tk_imbalance/results.md`。

## 基准（本机 202，C600UL 32 卡）

| TPR | 16 | 64 | 256 |
|---:|---:|---:|---:|
| full (µs/kernel) | 321.8 | 603.0 | 1800.7 |

详见 `test/70.r32_tpr_perf/results.md`。

## flag 结构体改造（local_flags / ranks_flags + current_time）

按参考实现 `mega/src/kernel/mega_fp8.cuh` 重构同步标志（2026-10-05，未提交 git）：

- 新增 `src/flags_defs.cuh`：
  - `struct alignas(256) local_flags { int64_t current_time[kGrid][64];
    uint32_t bar[1032]; int out_total[kGrid]; int out_processed[kGrid]; int dbg[8];
    uint32_t ck_dbg[kEPR*16]; unsigned long long pre_ticks[kGrid*16];
    float l1_chk[kGrid]; float l2_chk[kGrid]; }`（普通显存，本 rank 私有）；
  - `struct alignas(256) ranks_flags { uint32_t xbar_barrier[kGrid][kGrid];
    uint32_t combine_sums[kTopk][kMaxCombineRows][kCombineSlicesMax]; }`
    （FG 显存，跨 rank peer 可写；kMaxCombineRows=512、kCombineSlicesMax=32）。
- `MegaArgs` / `MegaLaunchArgs` 的 `bar_flags` 改 `local_flags* lf`，
  `epoch_ctr/last_release` 由每 block 单调 launch 计数 `lf->current_time[b][*]` 取代；
  xbar 跨 rank 旗从 sym 迁到 `rf->xbar_barrier`。
- 第二轮（本提交）：把控制/诊断缓冲 `out_total/out_processed/dbg/ck_dbg/
  pre_ticks/l1_chk/l2_chk` 从 `MegaArgs`/`MegaLaunchArgs` 独立指针/独立 mcMalloc
  收进 `local_flags`，kernel/host 读写一律走 `lf->...`；删除已无引用的
  `local_flags.last_ready`。跨 rank COMBINE 完成标记 `combine_sums` 从 sym 的
  `combine_sums_off` 区迁入 `ranks_flags::combine_sums[k][token][slice]`，
  COMBINE 推送/标记与 REDUCE 等待读取改经 `rf_peers[dst]->combine_sums`；
  `combine_rows`（BF16 数据）仍留在 sym，sym 分配随之去掉 sums 区。
- 行为不变：`bar[]` 索引逐一对齐原 `kBar*Idx`；可见性顺序（payload→fence→ready）
  与自旋/超时语义保持；`MX_DBG_COUNT=0` 默认零开销路径不变。

正确性（旧口径 R=2/32，`--perf-only 0 --compute 1`）全部 `[总体] PASS`；
性能（test/70，graph，2 次采样）：TPR=16/64/256 = 335~340/646~648/1909~1911
µs/kernel，相对基线 345~353/637~654/1901~1915 全在 ±3% 内（见 `perf.md` v3）。
test/71 不均衡 1788/3483/4749µs vs 基线 1776~1810/3458~3502/4686~4717，全在 3% 内。

## flag K 片轮转（M_CYCLE，对齐 mega_fp8.cuh）

时间标签 `[261005_22_06]`（未提交 git）。引入 `constexpr int M_CYCLE = 4`：
`local_flags::bar` 加 `[M_CYCLE]` 旋转维；`ranks_flags::xbar_barrier` 与
`combine_sums` 同理（后者 +1.5MB/rank，FG 显存）。每相位按单调 launch 计数
`current_time[b][*]` 派生 `c=ct%M_CYCLE`，只用第 c 片；内核入口对
`ct2=(c+2)%M_CYCLE` 槽 device 侧预清（graph replay 安全）。REDUCE 独立 kernel
由 `current_time[0][1]-1` 派生同一 c。可见性顺序与自旋/超时语义不变。
性能：test/70 TPR16/64/256 = 337.6~340.6/637.0~639.5/1904.1~1906.7µs（全在
基线 ±3% 内，TPR64 略快）；test/71 全档 1787/3448/4721µs（±3% 内）；R=2/32
全 `[总体] PASS`。

## 真实不均衡路由长尾优化（窗口切分 + COMBINE 标记修复）

时间标签 `[261005_23_23]`（未提交 git）。根因：`win_cap` 缺省被抬到
`max_tokens=512`，收到 512 行的专家 `ceil(512/512)=1` 个窗口 → 整段串行落在
单个 block（长尾 4.7ms）。另外 `win_cap<max_tokens` 时各 rank 的 `merge/
nslice_eff` 可不一致，而 COMBINE 只写 1 个物理 slice 的完成标记，源 rank 的
REDUCE 按自身 `nslice_eff` 等待会漏槽 → 4M 级自旋超时（多节点 graph 累积为
秒级“伪死锁”）。改动：

1. `combine.cuh`：`combine_push_pair` / `combine_mark_rows` 把物理任务覆盖的
   **全部逻辑 slice**（`slice*merge + [0,merge)`）标记写满（`merge=1` 逐字节等价）。
2. `bench.cu` `run_aligned`：逐测例检测长尾不均衡（`max_expert ≥ 4×非零中位数`）
   时把 `win_cap` 收缩到 `round16(cap_need/kRing)`（约 kRing 个窗口）；均衡分布
   保持 `max(128,max_tokens)` 不变。

性能：目标 `--tk-launch 853000`（max512）**4708.8 → ~1295µs（−72.5%，<2000）**；
804300（max362）3479→1030、804400（max176）1784→1289；均衡 test/70 三档
345/646/1915µs（±3% 内）。正确性 R=2 与 R=32 全 `[总体] PASS`（见 `perf.md` v5）。

### v6：长尾再优化（merge 不粗化，不均衡专项）

在 v5 基础上给 `MegaArgs`/`MegaLaunchArgs` 增加 `merge_grid`：不均衡长尾测例传大值
→ `merge=1`（不做 `nslice_eff` 粗化），并把窗口目标调到 45 行；大专家 512 行切成
12 窗×4 片的小任务摊满 32 block。均衡测例保持原粗化（零回归）。
目标 853000 **1295 → ~1034µs（再降 20.2%）**；804400=967、804300=877。正确性
（含强制 merge=1）R=2/R=32 全 `[总体] PASS`（见 `perf.md` v6）。

## 块级代价感知 LPT（`-s2` 不均衡专项，默认关）

新增编译开关 `R19_BLOCK_LPT`（默认 0，`-DR19_BLOCK_LPT=1` 开启）。动机：`-s2`
每 rank 内各 block 的 L1+L2 GEMM 代价（∝ 窗口行数）差异大，而现有 FCFS
（`atomicAdd(task_counter)`）只均衡任务「个数」。开启后 `dispatch_pull_body` 在
tid0 建表阶段对 `raw_tasks = s_cum[kEPR]*kNSlice` 个任务按代价
`cost=min(c_e-w*win_cap, win_cap)` 做**降序稳定计数排序**（smem `s_order`），
领号仍是原 `atomicAdd`，但 `task = s_order[counter]`（降序发放 + 动态领取 ≈ LPT）。
每任务解码 `(e,w,slice)`、`pool_slot`、ring 门、`l1_prog`/`l2_done` 协议完全不变
（`src/compute.cuh:679-726`, `:772-788`）。

- 正确性：开关开/关均 `check --all` **762/762 PASS**（含 win=32..128 自适应窗）。
- 性能（R=32, `-s1 tprs 32/128/256` 与 `-s2 top50 replays=3`，两次均值）：
  | 指标 | OFF(基线) | ON | 变化 |
  |---|---:|---:|---:|
  | s1 TPR32 | 488.8 | 496.4 | −1.6% |
  | s1 TPR128 | 944.6 | 964.6 | −2.1% |
  | s1 TPR256 | 1673.4 | 1692.9 | −1.2% |
  | s2 top50 mean | 765.2 | 766.5 | ±0.2%（中性） |
- 结论：**未达 ~400µs 目标**。根因是**任务量化**：s2 每 rank 重任务切片数常为
  32 的非整数倍（如 rank28 有 44 个满窗切片 / 32 block → 鸽笼必有 12 block 做
  2 个满窗），LPT 只能压低较空 block、无法降低最忙 block 的上界，故墙时不变。
  最忙 block 的 L1 max/min 由基线 2.51× 变为 2.71×（未收敛）。真正要降需**更细
  粒度派发**（每窗按 `kNK` 切 K 段，`raw_tasks = s_cum*kNSlice*kNK`）后再 LPT。

## `-s2` 细粒度派发实验（NS=8 与跨 block K 切，均默认关、零行为变更）

时间标签 `[261012_03_04]`。承接上节：直接尝试两种「更细粒度」派发，**结论均为
净回退**（未达 ~400µs），开关默认关、关时逐字等价基线。改动仅
`src/compute.cuh`、`src/compute_big.cuh`。

### Stage 1：N 切 NS=8（`-DKNSLICE_OVERRIDE=8` ± `-DR19_BLOCK_LPT=1`）

- 原理：把每窗任务数由 4 增到 8（N 组更细）。每片 A 重读 ×N、权重不重读、无需归约。
- 实测（R=32，`-s1 tprs 32/128/256 -nodes 3 -replays 2`；`-s2 top50 -replays 3`）：

  | 指标 | 基线 NS=4 | NS=8 | NS=8+LPT |
  |---|---:|---:|---:|
  | s1 TPR32 | 485.4 | 743.6 | 762.9 |
  | s1 TPR128 | 946.1 | 1208.8 | 1225.0 |
  | s1 TPR256 | 1678.6 | 2239.6 | 2277.8 |
  | s2 top50 mean | 767.9 | 1086.8 | — |

- **否决**：s1 回退 +28~53%、s2 回退 +42%。根因：N 切把 A 装载次数放大 NS 倍，
  且每窗协议任务数翻倍；「更细」的收益被访存放大吞掉。

### Stage 2：跨 block K 切（`-DR19_BLOCK_KSPLIT=1`，`R19_BLOCK_KSPLIT_WINMAX=64`）

- 实现（`src/compute.cuh` + `src/compute_big.cuh`）：把每个 `(e,w,slice)` 的 L1 沿
  K=H 再切 `kNK` 段、跨 block 派发（`raw_tasks = s_cum[kEPR]*kNSlice*kNK`，
  仅在 `win_cap ≤ 64`（TK/长尾小窗）+ fragment 可用时启用；大窗走原路径）。
  - **owner（kseg==0）**执行 pull/权重 staging/分片旗、等 `l1_prog`、跑 L2/tail、
    发布 `l2_done`；其余段只算 K 部分和 —— 严禁每段跑协议。
  - 各段把 gate/up 原值以 **FP32 atomicAdd** 累加到 `l1_partial[slot][slice]`；
    owner 在置分片旗前把该部分和缓冲与 `l1_part_cnt[slot][slice]` 清零（每代一次）。
  - `l1_part_cnt` 的 atomicAdd 判「最后一段」（`old==kNK-1`）→ `r19_l1_reduce`
    （SwiGLU→FP8→ffn_hidden/frag2）→ 由该段发布 `l1_prog`。
  - 新增函数：`r19_l1_ksplit_iso<kMT>`（`compute_big.cuh:556-625` 区）、
    `r19_l1_wait_all`（owner 只等不发布）；`r19_pull_flags` 增加 `kseg` 参数
    （非 owner 跳过 pull、只等分片旗）；任务体 `r19_big/small_task` 增加 `kseg` 并
    按 owner 决定 L2/tail。
- 正确性：开/关均 `check --all` **762/762 PASS**；`check --smoke` **8/8 PASS**
  （K 切开启 <10s）。
- 性能（R=32，同口径）：

  | `kNK` | s2 top50 mean | 相对基线 |
  |---:|---:|---:|
  | 1（基线） | 767.9 | — |
  | 2 | 1003.6 | +31% |
  | 4 | 1419.1 | +85% |
  | 8 | 1970.2 | +157% |

  s1（`kNK=8`）：TPR32 = 1246.8（+157%）、TPR128 = 979.1（win=128 未启用 K 切，
  ≈基线）、TPR256 = 1690.5（≈基线）。
- **否决**：K 切随段数单调变差。根因（实测 + 分析）：
  1. **每窗任务数 = NS×NK 把窗口「独占」整张 grid**（32 任务/窗），窗口由基线
     「8 窗并行」退化为「1 窗串行」；窗内 pull→各段→reduce→L2 的串行段无法被
     其它窗的延迟掩盖。`kNK` 越小任务/窗越少、并行窗越多，回退越小 —— 但始终为负。
  2. **FP32 部分和 atomicAdd 流量**：每 `(e,w,slice)` 写 `count×kL1SliceCols×4B`
     ×`kNK` 次（另加清零/读回）。小窗（`count≤48`）下这是不可忽略的 HBM 原子流量。
  3. 块级 L1 max/min：804900 rank28 由 K 切的 2.22×（min 212k / max 472k）
     **未收敛到目标 ≲1.2×**：段代价仍 ∝ 窗口行数，且 FCFS 按窗连续发牌。
- 结论：**未接近 400µs**；两阶段均默认关（`R19_BLOCK_KSPLIT=0` 编译期消除，
  关时 `check --all` 762/762、s1/s2 与基线逐字一致）。

## 方向1：B（权重）smem 异步双缓冲（`-DR24_BSMEM_B=1`，默认关）

时间标签 `[261012_09_40]`。动机：消融 `R24_DIAG_NOB` 显示 B 装载裸露 ≈211µs
（占 `-s2 top50` 的 28%）。实现：`r24_l1_big`/`r24_l2_big` 增加编译期 `kBsmB`
环深；B 经 `__builtin_mxc_ldg_b128_bsm` 搬入 smem 环、`r24_lds_b` 读，与 A 的 bsm
流水对齐。**关键坑**：该 builtin 的拷贝计入 **gvmcnt**（非 bsmcnt！）——`arrive(4096)`
等不到 B，必须 `arrive(64+N)=gvmcnt(N)`；A 可见性仍由 stage 顶 `bsmcnt(0)+__syncthreads`
保证。smem：默认保持 A 双缓冲 64KB 不变，开启时扁平数组放大到 112KB（A 48KB+B 64KB）。
```sh
scripts/build.sh so check mybench           # 默认关：逐字等价基线
KERNEL_CFLAGS="-DR24_BSMEM_B=1" scripts/build.sh so check mybench
```
- 正确性：关 `check --all` **762/762**；开 `check --all` **762/762**；`check --smoke` 均
  **8/8 PASS**（<10s）。
- 性能：s2 top50 **749.5→763.1µs（+1.8%）**；s1 TPR 32/128/256 = 483.5/967.9/1681.0
  （基线 482.1/947.4/1686.2）。**净回退，默认关**。
- 诊断：B-smem 把 `NOB` 暴露从 **211µs 降到 93µs**（延迟确实被隐藏 ~118µs），但
  `NOB` 地板由 544→677µs（+133µs）——B 经 smem 多一遍写+读，占用 smem 端口带宽。
  B 实为**带宽受限**（≈191µs@1.4TB/s），隐藏延迟不减少字节数。详见 `perf.md`。

## 方向3：通信/REDUCE 与计算重叠（`-DMXK_FUSE_REDUCE=1`，默认关，净中性/微回退）

时间标签 `[261012_11_10]`。承接 `-s2 top50` ~751µs → 400µs 目标，尝试「把通信/REDUCE
与计算重叠」。先把「通信地板」逐项消融量化（R=32，`us_mean`）：

| 消融（`-s2 top50`） | mean (µs) | 边际成本 |
|---|---:|---:|
| 基线 | 753.3 | — |
| `CM_SKIP_GEMM=1`（纯通信+REDUCE） | 174.9 | 地板（**非**与 GEMM 相加项） |
| `MX_SKIP_REDUCE=1` | 739.0 | REDUCE ≈ 14.3 |
| `PULL_LOCAL_DIAG=1` | 740.4 | A 远读 ≈ 13.0 |
| `R30_DIAG_NOPUSH=1` | 805.7 | 推送已重叠（去之反而更慢） |

**关键结论**：`CM_SKIP_GEMM` 的 174.9µs 地板**不等于可回收量**——A pull / COMBINE
推送 / 跨 block wait-all 大部分已被 GEMM 掩盖；s2 真正暴露的通信边际只有
REDUCE≈14µs + A 远读≈13µs ≈ **27µs（3.6%）**，方向3 上限约 749→722µs。s1 主要
暴露 A 远读 45~65µs（`PULL_LOCAL_DIAG`：TPR128 949.4→884.9、TPR256 1670.2→1625.4）。

候选1 实现（`MXK_FUSE_REDUCE`）：`reduce_body` 拆出 `reduce_tile_body`
（`src/combine.cuh:481`）+ wrapper；`mega_kernel` 任务循环后就地按 `(token,half)`
轮分执行（`src/kernel.cuh`）；`launch_full` 不再单发 `mega_reduce_kernel`
（`src/megakernel.cu:231`）；host 经新增 `MegaArgs::y`（`src/compute.cuh:78`）传输出。
```sh
scripts/build.sh so check mybench                         # 默认关：逐字等价基线
KERNEL_CFLAGS="-DMXK_FUSE_REDUCE=1" scripts/build.sh so check mybench
```
- 正确性：关 `check --all` **762/762**（78.6s）；开 `check --smoke` **8/8 PASS**（<10s）。
- 性能：s2 = 749.1→**755.0µs（+0.8%）**；s1 TPR32 485.2→492.8（+1.6%）、
  TPR128 950.5→983.8（+3.5%）、TPR256 1675.5→1717.1（+2.5%）。**净中性/微回退，默认关**。
- 根因：`MT_DUMP_RES=1` 显示 `full regs=256 smem=66160`——SM/寄存器已被 mega 占满，
  reduce 无并发窗口；且融合把 REDUCE 并行度由 `2*tokens` block 降到 `kGrid=32`
  （s1 大档降 16×）。跨 launch side-stream 重叠亦被 harness 阻断：本机
  `BeginCapture(ThreadLocal)` 下未 join 的侧流令 `EndCapture` 返回 904 且无图产出。

详见 `perf.md` [261012_11_10]。

## 方向4：win=512 + 跨 block 扁平池 NS4×NK8 L1 split-K（`-DR19_BLOCK_KSPLIT=1
-DR19_BLOCK_KSPLIT_WINMAX=512`，默认关，净回退）

时间标签 `[261012_13_55]`。目标 `-s2 top50` 734µs → 逼近 400µs：假设 win=512（每专家
1 窗）消除按窗重复读权重，再用 `(e, N片 s, K段 c)` 共 32 个等代价子任务填入同一扁平池
恢复负载均衡。**已用逐块相位诊断核实这是真正的块级派发**（非块内 K 循环）：

| 口径（s2 top50 mean, µs） | 默认(自适应win) | win=512 无切 | win=512 + NS4×NK8 |
|---|---:|---:|---:|
| 完整 | **733.9** | 1992.1 | 2381.7 |
| 纯通信+协议（`CM_SKIP_GEMM=1`） | 171.2 | 486.9 | — |

- **块级分布**（rank17, case 806200）：win=512 无切仅 **12/32** block 有 L1
  （热专家 4 个 N 片任务独占 b=24..27，L1=2.33M cyc，`max/min=7.66`）；加 NS4×NK8 后
  **32/32** block 有 L1（每块 5~8 任务，`max/min=1.46`）。均衡达成，但每块 L1 均值
  1.33M vs win=64 的 ~0.4M → **绝对耗时 ×3.3**。
- **根因**：K 切把「M（窗）并行」换成「K 并行」，A-smem 双缓冲只剩 3 段流水
  （`nstage=chunk/kKS`），每任务固定开销（流水未热身 + B 直装 + 协议）不随 K 缩小；
  本机权重**非瓶颈**（`CM_SKIP_GEMM` 仅 171µs），win=512 省下的权重重复读 < 失去并行的代价。
- **归约非瓶颈**：win512 base（无 atomicAdd）≈ ksplit；`r19_l1_reduce` 仅 ~227µs。
  「非重叠 `l1_partial[e][s][c]`」需 ×NK 缓冲且末段 reduce 读量 ×NK，**更慢，未采用**。
- 门禁：默认关 `check --all` **762/762**（75.4s）；开 `check --all` **762/762**（134.8s）、
   `--smoke` 8/8。**未接近 400µs**；两开关保持默认关。详见 `perf.md` [261012_13_55]。

## 方向5：fp4(e2m1) 权重 + packed 预取环（`-DMX_FP4_W=1`，默认关）

时间标签 `[261012_18_58]`。目标：W1/W2 以 fp4 打包存储（HBM 字节减半），kernel 用
`__builtin_mxc_cvt_pk8_f4tof8_scale_lo` 解包回 fp8 fragment 复用现有 fp8 MMA。

- **原型问题**：把 cvt 放进 B 预取环（`load→cvt` 即刻产出 fp8 fragment）时，
  `-s2 top50` 从 695µs 恶化到 877µs（B 暴露 216→427µs）。微基准（同会话，本机
  C600-UL 32SM）显示该 cvt 为**延迟受限**：128×256 时 362 Gcvt/s，32×256（=本 kernel
  的 4 warps/SM）仅 123 Gcvt/s。`MX_FP4_NOCVT` 隔离实验（只删 cvt，留 8B 装载）
  得 660µs < 基线 → **代价全在 cvt 的调度暴露，不在字节/装载宽度**。
- **修法（`MX_FP4_RING`，默认 1）**：预取环只缓存 **8B packed fp4**（寄存器减半），
  解包推迟到消费点，并按 k-step 批量展开 `bexp[kBatch]` 后再喂 MMA，给编译器更大
  调度窗口。应用到 r19/r24 的 L1/L2 四条 GEMM 路径。
- **A/B（同会话，us_mean）**：

| 口径 | 基线 | fp4+RING | 变化 |
|---|---:|---:|---:|
| `-s2 top50` mean（3 轮） | 698.2 | **581.6** | **−16.7%** |
| `-s2` B 暴露（`R24_DIAG_NOB`） | 216.4 | **134.0** | **−38%** |
| s1 TPR=32 | 446.9 | 396.9 | −11.2% |
| s1 TPR=64 | 620.1 | **494.2** | **−20.3%** |
| s1 TPR=128 | 915.3 | 890.9 | −2.7% |
| s1 TPR=256 | 1597.3 | 1562.5 | −2.2% |

- 门禁：`check --smoke` **PASS 8/8**，6.4s（<10s）；默认（`MX_FP4_W=0`）逐字等价基线。
- 遗留：距 500µs 仍差 ~82µs；`MX_FP4_W=1 + R24_BPF=4 + R19_L1_PF=8` 可再降至 ~567µs，
  但需做成 **fp4 条件默认**（否则回退默认 fp8 路径，历史 `-s2` TPR128/256 回退），未纳入。
  权重 fp4 量化/reorg 在 torch 侧尚未接入（本环境无 torch）。

## torch 层（pybind + Python 包，可安装 wheel）

时间标签 `[261006_12_40]`（未提交 git）。新增 `torch/`：pybind11 扩展
`mxmegakernel._C` + Python 包 `mxmegakernel`，对齐 sglang megamoe 后端
（C600UL fp8xfp8）所需的最小 API 面，并真实绑定 mini_mega kernel。

| 能力 | 状态 |
|---|---|
| `get_token_alignment_for_mega_moe` | 已实现 |
| `select_fp8_mega_moe_window`（FP8 AUTO 表） | 已实现 |
| `get_symm_buffer_size_for_mega_moe` / 布局偏移 | 已实现 |
| `transform_weights_for_mega_moe`（fp8xfp8） | 已实现（与源项目逐字节等价） |
| `SymmBuffer` 多 rank FG/IPC + 建档预热 + 输入视图 | 已实现 |
| `_C.mega_launch_empty` / `fill_weights_lane` | 已实现（真实 kernel 绑定） |
| `fp8_mega_moe` 融合发射（`fp8mega::launch_full`） | 已实现 |
| `bf16/fp8_fp4` 路线、`transform_sf_into_required_layout` | not_implemented |

构建：`cd torch && MACA_PATH=/opt/maca python setup.py bdist_wheel`
→ `torch/dist/mxmegakernel-0.0.0+mini.mega.20261006b-cp310-cp310-linux_x86_64.whl`。

数值自检：`tests/smoke.py`（单卡）→ `SMOKE OK`；
`torchrun --nproc_per_node=32 tests/test_fp8_mega_moe.py --ranks 32`（71，真实
R=32/E=256 几何，随机权重 vs 参考）→ 32/32 PASS；本地 R=16 window=64 亦全过。

71 跑通：GLM-5.2-FP8（tp32/dp32/megamoe/fp8）服务 ready（`/health=200`，
`max_total_num_tokens=148160`），`/generate` 输出连贯文本；并发压测
48 请求 / 并发 8：`output_throughput=114.5 tok/s`。需要三个 sglang 侧补丁
（权重原位接管、SwiGLU 缩放契约修复、预热），详见 `torch/README_torch.md`
与 `torch/sglang_patches/`。
