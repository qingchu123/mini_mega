# R=32 性能基准（perf.md）

本文件记录 **R=32** 各 TPR 的性能基准，并在每次优化后追加一列（版本列 + 该版本数据 + 相对基线提升）。

## 测试口径
```
MACA_VISIBLE_DEVICES=$(seq -s, 0 31) timeout -k 3 15 ./bench \
  --ranks 32 --nodes 20 --graph 0 --perf-only 1 --compute 1 \
  --perf-pattern 0 --tokens <TPR> --max-tokens 512 --win-cap 128
```
- 单档运行 ≤3s（严格 `timeout -k 3`）；每档取 2 次均值。

> 现行口径（对齐 `mega/src/bench/bench_mega_fp8.cu`）：单进程 `run_aligned`，
> `./bench` 默认 R=32 TPR={1,4,16,32,33,64,128,256}、均衡、graph `nodes=20 replays=2`
> （取 min）、默认不校验，单次约 2.9s。上表 v0/v1 为改造前分档子进程口径，
> 改造后同档复测：TPR=16/64/256 = 343.5/651.0/1918.3µs（±1% 内，见 test/70）。
- `full` = 全链单 launch 平均耗时（µs）；正确性另以 `--perf-only 0` 校验，
  全 TPR 三行 `[数值校验] PASS` + `[总体] R=32 PASS`。
- win_cap **固定 128**（与 `--expert-window`/环境变量无关）。

## 版本记录
- **v0**：本优化会话前的基线（git HEAD `39a3aaf`，`src/{prologue,compute_big,combine}.cuh`）。
- **v1**（2026-09-28）：本轮优化——
  1. `prologue.cuh`：route_build 在 `R > 8` 时走全线程（小 pair 20% pad 路径）；
  2. `compute_big.cuh`：① count∈(48,64] 的 L1/L2 改走 r24 A-smem（MT4 无掩码），
     且 L2 的 N 批 2→3；② count∈(32,48] 档 L1 用 MT3、L2 用单次 MT3/3；
  3. `combine.cuh`：任务尾推送改 warp 连续行段。

## 基准表
| TPR | v0 基线 (µs) | v1 (µs) | v1 提升 |
|---:|---:|---:|---:|
| 1 | 292.5 | 294.3 | −0.6% |
| 2 | 300.3 | 304.7 | −1.5% |
| 4 | 310.0 | 310.8 | −0.3% |
| 8 | 323.4 | 323.9 | −0.2% |
| 16 | 342.9 | 345.9 | −0.9% |
| 17 | 414.3 | 408.3 | **+1.5%** |
| 32 | 475.9 | 462.9 | **+2.7%** |
| 33 | 560.6 | 555.4 | **+0.9%** |
| 64 | 744.5 | 655.1 | **+12.0%** |
| 65 | 741.4 | 742.7 | −0.2% |
| 128 | 1023.0 | 1027.1 | −0.4% |
| 129 | 1325.2 | 1331.5 | −0.5% |
| 256 | 1892.4 | 1914.8 | −1.2% |
| 257 | 2226.1 | 2203.6 | **+1.0%** |
| 512 | 3671.5 | 3697.6 | −0.7% |

> 结论：v1 在 TPR=64（−12.0%）、32（−2.7%）、17（+1.5%）、33（+0.9%）、257（+1.0%）
> 改善；其余 TPR 在 ±1.5% 内（属 code-layout collateral 与测量漂移，非功能回归）。
> 注：v0 的 TPR=128 实测 1023µs，与 v1 1027µs 基本一致——早前对照的
> “test/59 1007.7µs”属不同测量条件，并非本会话引入的回归。

## v2：flag 结构体改造（local_flags / ranks_flags + current_time）

时间标签：`[261005]`（未提交 git）。按参考实现 `mega/src/kernel/mega_fp8.cuh`
把扁平 flag 数组收敛为 `local_flags`（普通显存，本 rank 私有）+
`ranks_flags`（FG 显存，跨 rank peer 可写）；每 block 的 `current_time[b]`
单调 launch 计数取代 `epoch_ctr + last_release`，xbar 跨 rank 旗迁入
`ranks_flags::xbar_barrier`。`bar[]` 索引与原 `kBar*Idx` 逐一对应，可见性
顺序与超时语义不变。

同机同口径（test/70 graph，4 次采样取均值；基线为任务给定改前值）：

| TPR | 基线 (µs) | v2 (µs) | v2 相对基线 |
|---:|---:|---:|---:|
| 16 | 353 | 350.3 | **+0.76%** |
| 64 | 651 | 644.9 | **+0.94%** |
| 256 | 1915 | 1906.6 | **+0.44%** |

> 结论：三档均略快于基线（全在 ±3% 内，无性能下降）；正确性 R=2/8/32 全 `[总体] PASS`。

## v3：控制/诊断状态与 combine_sums 收进 flag 结构体

时间标签：`[261005]`（未提交 git）。在 v2 基础上继续收敛：
`out_total/out_processed/dbg/ck_dbg/pre_ticks/l1_chk/l2_chk` 从
`MegaArgs`/`MegaLaunchArgs` 的独立指针 + 独立 mcMalloc 收进 `local_flags`，
kernel/host 读写一律走 `lf->...`；删除已无引用的 `local_flags.last_ready`。
跨 rank COMBINE 完成标记 `combine_sums` 从 sym 区迁入
`ranks_flags::combine_sums[kTopk][kMaxCombineRows][kCombineSlicesMax]`
（=8×512×32×4=512KB/rank），COMBINE 推送/标记与 REDUCE 等待改经
`rf_peers[dst]->combine_sums`；`combine_rows`（BF16 数据）仍留 sym，
sym 分配随之去掉 sums 区，`sums_stride` 恒 = kCombineSlicesMax=32。

同机同口径（test/70 graph，2 次采样；基线为任务给定值）：

| TPR | 基线 (µs) | v3 (µs) | v3 相对基线 |
|---:|---:|---:|---:|
| 16 | 345~353 | 335.2 / 339.6 | −1.5% ~ +0.8% |
| 64 | 637~654 | 645.9 / 647.7 | −0.2% ~ +1.7% |
| 256 | 1901~1915 | 1909.4 / 1910.6 | −0.6% ~ +0.5% |

test/71 不均衡（TK_INFO）：均衡 347.2µs；max176/362/512 = 1788.2/3483.2/4749.1µs
vs 基线 346~355 / 1776~1810 / 3458~3502 / 4686~4717，全档在 ±3% 内。

> 结论：无性能回退（三档与不均衡全档均在 ±3% 内）；正确性 R=2 与 R=32 全 `[总体] PASS`。
> 未做（可选）：`route_slot_counts`/`expert_recv_counts` 小计数未搬入 `ranks_flags`。

## v4：flag K 片轮转（M_CYCLE）

时间标签：`[261005_22_06]`（未提交 git）。按参考实现 `mega/src/kernel/mega_fp8.cuh`
加入 `constexpr int M_CYCLE = 4` 与轮转维：

- `local_flags::bar` → `bar[M_CYCLE][kFp8BarFlagsNumInt32]`；
  `ranks_flags::xbar_barrier` → `[M_CYCLE][kGrid][kGrid]`；
  `ranks_flags::combine_sums` → `[M_CYCLE][kTopk][kMaxCombineRows][kCombineSlicesMax]`
  （+1.5MB/rank，FG 显存，可接受）。
- 每相位入口按 `current_time[b][0]/[b][1]` 的单调 launch 计数派生
  `c = ct % M_CYCLE`，把 `&lf->bar[c][0]` / `&rf->xbar_barrier[c][0][0]` /
  `&rf->combine_sums[c][0][0][0]` 作为本轮「当前 cycle 切片」传入现有同步函数，
  `flags.cuh` 的 `bar_flags + idx` 寻址逻辑不变。内核入口对
  `ct2=(c+2)%M_CYCLE` 槽做一次 device 侧预清（CUDA graph replay 安全）。
- `l2_done` 仍随 `bar[c]` 轮转：它在 TASK_SETUP 每轮被重置为 0、判据是
  launch 内相对计数，轮转不影响其目标推导（方案 a 的等价实现）。
- REDUCE 是独立 kernel：读 `current_time[0][1]-1` 派生同一 c，读取
  `lf->bar[c]` 与 `rf->combine_sums[c]`，graph 下无需 host 传参。

同机同口径（test/70 graph，4 次采样；基线为同会话改前）：全档在 ±3% 内，
其中 TPR=64 略快（−2.2%）。test/71 不均衡全档 1787/3448/4721µs vs 基线
1784/3479/4709，全在 ±3% 内；均衡复测 337µs（与基线 336.8 持平）。
正确性 R=2 与 R=32 全 `[总体] PASS`。

| 口径 | 基线 (µs) | v4 (µs) | v4 相对基线 |
|---|---:|---:|---:|
| test70 TPR16 | 342.8 | 337.6~340.6 | +0.6%~+1.5% |
| test70 TPR64 | 651.6 | 637.0~639.5 | **+1.9%~+2.2%** |
| test70 TPR256 | 1905.4 | 1904.1~1906.7 | ~0% |
| test71 均衡 | 338.6 | 337.1~346.8 | −2.4%~+0.4% |
| test71 max176 | 1784.2 | 1787.2 | −0.2% |
| test71 max362 | 3479.0 | 3448.4 | +0.9% |
| test71 max512 | 4708.8 | 4721.4 | −0.3% |

> 结论：M_CYCLE 轮转未引入性能回退，且为完成旗「无需每轮重置」提供了机制
> 保证；测试 R=2/R=32 全 PASS。

## v5：真实不均衡路由长尾（TK_INFO max512）— 窗口切分 + COMBINE 标记修复

时间标签：`[261005_23_23]`（未提交 git）。

- **根因**：`win_cap` 缺省被抬到 `max_tokens=512`，`ceil(512/512)=1` → 那个
  收到 512 行的专家只生成 **1 个窗口/任务**，整段 512 行串行落在单个 block，
  其余 block 只做 8 个小任务 → 长尾 4700µs（均衡仅 ~340µs）。
- **关键缺陷（win_cap<max_tokens 死锁/超时的根因）**：`merge`（[R35] 粗化）
  按各 rank 自身 `ceil(count/win_cap)` 推导，各 rank 的 `nslice_eff` 可不一致；
  COMBINE 只写 1 个物理 slice 的完成标记 `combine_sums[k][token][slice]`，
  而源 rank 的 REDUCE 按自身 `nslice_eff`（可达 4）等待 → 漏槽 → 每 launch
  4M 级自旋超时（`dbg[4]` 暴涨），多节点 graph 下累积成秒级“伪死锁”。
- **改动**：
  1. `combine.cuh`（`combine_push_pair` / `combine_mark_rows`）：物理任务把
     其覆盖的**全部逻辑 slice**（`slice*merge + [0,merge)`）标记写满，使任意
     源 rank 按自身 `nslice_eff` 等待都不漏槽。`merge=1` 时逐字节等价旧行为。
  2. `bench.cu` `run_aligned`：新增长尾不均衡判据（**逐测例** `max_expert
     ≥ 4×非零中位数`）时，把 `win_cap` 收缩到 `round16(cap_need/kRing)`
     （约 kRing 个窗口，实测 64 最优：正好 512/64=8=ring，无槽复用）；
     均衡分布保持 `max(128,max_tokens)` 不变（避免短任务被切碎回归）。

目标场景（`--tk-info 0928_tk_info.log --tk-launch 853000`，default nodes=20
replays=2）多次采样：

| 采样 | us/kernel |
|---:|---:|
| 1 | 1298.60 |
| 2 | 1294.53 |
| 3 | 1301.06 |
| 4 | 1291.97 |
| 5 | 1298.15 |

**4708.8 → ~1295µs（−72.5%，<2000 目标达成）**。

回归（同机同会话；基线为本文档改前值）：

| 口径 | 基线 (µs) | v5 (µs) | v5 相对基线 |
|---|---:|---:|---:|
| test70 TPR16 | 342.8 | 345.1 / 345.3 | −0.7% |
| test70 TPR64 | 651.6 | 646.1 / 640.1 | **+0.8%~+1.8%** |
| test70 TPR256 | 1905.4 | 1913.6 / 1915.1 | −0.4% |
| test71 均衡 | 338.6 | 343.9 | −1.6% |
| test71 max176 | 1784.2 | 1289.4 | **+27.7%** |
| test71 max362 | 3479.0 | 1029.6 | **+70.4%** |
| test71 max512 | 4708.8 | 1296.0 | **+72.5%** |

> 结论：目标 853000 由 4700µs 压到 ~1295µs；另两个不均衡档同幅改善；
> 均衡 test/70 三档全在 ±3% 内（TPR16/256 与基线持平，TPR64 略快）。
> 正确性：R=2 与 R=32 均 `[总体] PASS`（三形态 + L1/L2/REDUCE 三行数值 PASS）。

## v6：长尾再优化 —— merge 不粗化（不均衡专项）— 853000 再降 ~20%

时间标签：`[261005_23_xx]`（未提交 git）。在 v5（win_cap 收缩）基础上：

- **根因**：v5 后大专家被切成 ~kRing 个窗口，但 `compute.cuh` 的 `merge` 粗化
  仍把 `raw_tasks`（窗数×kNSlice）合并到 ≤ 单波（nslice_eff 变小），使每个物理
  任务覆盖多个 slice、粒度偏大，长尾仍未摊满 grid。
- **改动**：给 `MegaArgs`/`MegaLaunchArgs` 增加 `merge_grid`（默认 = grid，保持
  原粗化）；bench 对「长尾不均衡」测例传一个很大的值 → `merge=1`（不粗化），
  同时 `win_cap` 目标从 64 调到 **45**。大专家 512 行 → 12 窗 ×4 片 = 48 个
  16~45 行小任务，摊到全部 32 block。对均衡测例不改（仍走原粗化，零回归）。
- **正确性**：强制 `merge=1` 下 R=2 与 R=32 三形态 + L1/L2/REDUCE 数值全 PASS
  （已验证后再把默认改回 grid）。

同机同会话（853000，5 次采样）：

| 采样 | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| us/kernel | 1038.32 | 1035.99 | 1033.43 | 1033.45 | 1031.12 |

**1295 → ~1034µs（再降 20.2%）**；相对 v5 前 4700µs 累计 −78%。

回归（test/70，µs）：TPR16/64/256 = 341~343 / 641~653 / 1921（基线内 ±3%）；
test/71：均衡 346、max176=967、max362=877、max512=1036（均优于 v5）。
正确性：R=2 与 R=32 全 `[总体] PASS`。

## 追加方式（后续每次优化）
在表尾新增一列：`| vN (µs) | vN 提升 |`，其中 `vN 提升 = (v0 − vN) / v0`，
并在「版本记录」追加一行说明该版本改动。示例（v2 占位，勿填入未测数据）：

```
| TPR | v0 基线 (µs) | v1 (µs) | v1 提升 | v2 (µs) | v2 提升 |
```

## 相关记录
- `test/69.r32_small_count_opt/results.md`：本轮改动明细与负结果。
- `analysis/17.r32_tpr_sweep/results.md`：全档扫描与台阶（M-tile 粒度）归因。
- `analysis/16.r32_small_count_opt/results.md`：瓶颈归因与后续方向。

## R=8 基准

### 测试口径
```
MACA_VISIBLE_DEVICES=$(seq -s, 0 7) timeout -k 3 15 ./bench \
  --ranks 8 --nodes 20 --graph 0 --perf-only 0 --compute 1 \
  --perf-pattern 0 --tokens <TPR> --max-tokens 384 --win-cap 128
```
- 单档运行 ≤15s（严格 `timeout -k 3 15`）；本次为单次运行（full，单位 µs，来自 `/tmp/r8_*.log`）。
- `--perf-only 0` 开启校验，全 TPR 均 `[数值校验]×3 PASS` + `[总体] R=8 PASS`。
- win_cap **固定 128**。
- 注意：bench 仅支持 `--ranks 2|8|32`（**不支持 16**）；本次为 `src/` 骨架 bench（不含 torch 侧改动）。

### 基准表
| TPR | full (µs) |
|---:|---:|
| 1 | 283.635 |
| 8 | 309.069 |
| 16 | 326.042 |
| 32 | 433.254 |
| 64 | 571.469 |
| 128 | 916.595 |
| 256 | 1653.107 |
| 384 | 2421.427 |

> 结论：R=8 全档随 TPR 单调上升，1→384 由 283.6µs 增至 2421.4µs；
> TPR≥16 后进入近似线性区间（每档约 +100µs/16 TPR 量级），符合计算/通信随 token 数扩展的预期。
> 全档数值校验与总体校验均 PASS。

## [261006_02_40] 不均衡长尾专项：多窗口 A fragment 布局（未提交 git）

### 背景与根因
- 目标场景 `--tk-info ... --tk-launch 853000`（R=32，max_expert=512，win_cap=45）基线 ~1034µs。
- 消融发现 **A（激活）装载** 与 B（权重）同为主成本：把 A 装载置零（`CM_DIAG_NOA`）
  1034→818µs（约 −210µs）。原因：该场景每专家多窗口 → `afrag_ok=false`，
  A 走**行主序散射装载**（每条 warp 指令跨 16 行 × 32B = 16 cache line）；
  单窗口场景才走已交付的 fragment（4 line）快路径。
- B 侧：寄存器预取加深会溢出（256 reg 顶格）；smem 异步分级流水实测更慢
  （gvmcnt 在飞级数受限 + LDS 往返），放弃。

### 改动要点
1. `compute.cuh`：`l1_gemm_swiglu_body_paired` / `l2_gemm_body` 增加**编译期**
   `bool kUseFrag` 模板参 + A fragment 装载（`cm_load_a_frag`）与 L1 输出
   fragment 写（`cm_store_ffn_frag`）。**注意：运行时 `fbase!=nullptr` 分支会
   破坏热路径调度（实测 3× 回退），必须编译期选择。**
2. `compute_big.cuh`：多窗口 `frag_all` 路径——pull 按 `(expert, w%ring)` 槽
   同时写行主序 pool（bench 校验用）与 A fragment；L1 读 A fragment、写 F2
   fragment；L2 读 F2 fragment。
3. `bench.cu`：仅不均衡长尾场景分配 A/F2 fragment 专用缓冲（`[EPR*ring]` 槽），
   并启用 fragment；**均衡场景不分配 → 走精简内核，避免 collateral**。
4. `kernel.cuh`/`megakernel.cu`：`mega_kernel` 增加 `kFragCap` 模板参，host 按
   `frag_a != nullptr` 选择 fragment 实例（均衡 TPR 测例不受代码膨胀影响）。
5. host 自动 win_cap：不均衡场景由 45 抬到 **48**（fragment 消除 A 散射惩罚后，
   较大窗口减少 B 重复装载且不再有 A 惩罚）。

### 结果（同机多次采样，R=32）
| 场景 | 基线 (µs) | 优化后 (µs) | 提升 |
|---|---:|---:|---:|
| 853000（目标） | 1032~1040 | **813~819** | **−21%** |
| 804400（max176） | ~968 | 808 | −16% |
| 804300（max362） | ~878 | 727 | −17% |
| test/70 TPR16 | 341~346 | 344~351 | +0.5~1.5% |
| test/70 TPR64 | 641~653 | 653~666 | +1~2% |
| test/70 TPR256 | 1921 | 1952~1956 | +1.6~1.8% |

### 正确性
- R=2 / R=32 `[总体] PASS`（L1/L2/REDUCE 数值三行 PASS）。
- fragment 路径与行主序路径 **bit-exact**（同 win_cap=48 下 l1_chk/l2_chk 累加和逐位相同）。

## v7：tk_log 前 50 launch 平均优化（并行 4+2 agent；接受 −7%）

- 口径：`./bench --tk-info <log> --tk-first 50`（升序 804300..809200，n=50 平均）。
- 目标：−20%（898.5 → 718.8µs）；**实际 −7.0%（898.5 → 835.8µs）**。
- 合入：
  - 逐例自适应 win_cap（消除全局聚合被离群例抬高）：855.4（−4.8%，eb8774b）；
  - 中等档 count∈33..48 的 L1/L2 改走 r24 A-smem + bsm 异步：835.8（再 −2.3%，cce2fdf）。
- 负结果（已回退/默认关）：远写 push dispatch（+4.0%，核内远端读已被计算隐藏）；
  跨任务软件流水（pull/wait-all 已隐藏，≤4%）；B 权重重读减量（暴露 ~247µs，但减 B 须减窗口，
  与长尾均衡强耦合；win_cap=48 已最优）。
- 卡点：目标 718.8µs 低于 Bfix 地板（B 地址 L1 常驻、指令保留）≈745µs；进一步需
  **多级 async smem B 深流水 / B-stationary** 这类高风险重构（现有 B-smem 因
  `__mxc_arrive(4096)` 一次排空而失效）。
- 回归：test/70、test/71 均 <3%；正确性 R=2/R=32 `[总体] PASS`。

## v8：tk_first 前 50 launch：编译期裁掉不可达的 MT8 大档（kCap64）

时间标签：`[261007_00_57]`（分支 agent/opt10h1/1，commit 见下）。

- 口径：`./bench --tk-info test/70.r32_tpr_perf/0928_tk_info.log --tk-first 50`
  （升序 804300..809200，n=50 平均）。
- 根因（探针）：tk 场景 win_cap 逐例自适应后恒 ≤64（多数 48，806600 为 64），
  因此单个任务 `count = min(remain, win_cap) ≤ 64`，`count>64` 的 **MT8 异步
  大档（r24_l1_big/r24_l2_big<MT8>）永不可达**；但该档的 `acc[kBatch][MT8][kNT][4]`
  与内联主体仍参与 `r19_big_task`（`__noinline__`）的寄存器/栈分配，把同函数内
  `count≤64` 的小/中档路径一并拖慢。
- 改动：新增编译期模板参 `kCap64`（`mega_kernel`→`mega_dispatch_loop`→
  `dispatch_pull_body`→`r19_big_task`）。`win_cap<=64` 且宿主提供 fragment 缓冲
  （不均衡/tk）时选用该实例；`else if constexpr (kCap64)` 直接落到固定 MT4 回退，
  **保证不实例化 MT8 模板**（不占栈）。均衡 test/70（无 frag）与 TPR=256 大档
  仍走原实例，零影响。
- 前 50 launch 平均 **835.8 → 809.9µs（-3.1%）**（多次 810.6/809.9/810.8）。
- 回归：test/70 TPR16/64/256 = 345~354 / 661~668 / 1930~1935µs（基线
  346~353 / 657~661 / 1946~1956，全在 ±3% 内，TPR=256 反降）；test/71
  均衡 343.7、804300=697、804400=763、853000=750µs（均不劣于基线）。
- 正确性：R=2 `[总体] PASS`（L1/L2/REDUCE 三行数值 PASS）。
- 负结果（本会话，已回退/默认关）：
  - B smem 异步分级流水（`CM_SMALL_BSM`/`CM_USE_BSM`，并修正原 A 预取
    「先写后读」串行 bug）→ tk 前 50 中性/略负（831~844µs）；微基准证实
    `arrive(64+N)` 分级语义正确，B 非纯延迟受限。
  - 全局 `R19_L1_MT=R19_L2_MT=4`（tk -4.1%）但 TPR=256 **+13%**，不可接受；
    故改为仅在不可达的 cap64 实例上删档。
  - 加大 win_cap / kRing / kNSlice / 寄存器 B 预取深度（L1PF/L2PF/BPF）均回退。

> 目标 700µs 仍未达；tk_first 剩余 ~810µs 的暴露主要来自 B（权重）HBM 流量
> （BFIX 诊断地板 ~542µs，n=0 变址 ~587µs），需 B 复用/L2 驻留类重构。

## v9：tk_first 前 50 launch：kCap 上界再下探到 48，裁掉 MT4 大 GEMM 兜底

时间标签：`[261007_02_29]`（分支 agent/opt10h/1）。

- 口径：`./bench --tk-info ... --tk-first 50`（n=50 平均）。
- 阶段消融（同会话，`-DCM_SKIP_L1/L2`）：full=807.8、comm=201.9、
  L1=379.6（46.9%）、L2=241.5（29.9%）；L1 最大 → 优先攻 L1。
- 根因：v8 的 kCap64 只删了不可达的 MT8 异步档；但 tk 逐例 win_cap 多数=48，
  此时 count>48 全部不可达，`r19_big_task` 里 **MT4 大 GEMM 兜底
  `r19_l1_big<4,R19_L1_PF,4>`（acc 4×4×2×4=128 floats）与
  `r19_l2_big<4,...>`** 仍被实例化并抬高寄存器/栈。
- 改动：把 v8 的 `bool kCap64` 泛化为编译期 `int kCap`（0/48/64）。win_cap<=48
  选 `kCap=48` 实例：`else if constexpr (kCap48)` 用**已实例化的 paired MT3**
  兜底替换两个大 GEMM，`if constexpr` 保证不实例化 `r19_l1_big/r19_l2_big`。
  该兜底对生产 g_l1=4/g_l2=6 几何不可达（正确性不受影响）。
- 前 50 平均 **810.7 → 776.1~778.2µs（-4.0%，多次）**；相对 v7 835.8 累计 -7.1%。
- 回归：test/70 TPR16/64/256 = 347~351 / 649 / 1932~1935（基线内）；
  test/71 均衡 349、804300=672、804400=724、853000=711（均优于 v8）。
- 正确性：R=2 `[总体] PASS`（L1/L2/REDUCE 三行 PASS）。

## v10：tk_first 前 50 launch：跨 graph 节点前序复用 + tk 专属 B 预取加深

时间标签：`[261007_10_23]`（分支 agent/opt700/1）。

- 口径：`./bench --tk-info test/70.r32_tpr_perf/0928_tk_info.log --tk-first 50`
  （升序 804300..809200，n=50 平均）；基线 v9 = 776.8µs。
- 根因（消融，同会话）：
  - `CM_SKIP_GEMM`（纯通信）=201.8µs；进一步 `R19_DIAG_NOPULL` 只降 8µs →
    纯通信几乎不含 A 拉取数据量，主要是「前序 + flag/同步 + 归约门」。
  - `R30_DIAG_NOTPUSH`（任务尾 combine 数据推送置零）单测降 ~39µs，但推送为
    协议必需；把推送融合进 L2（paired/小 r24 路径）实测更慢（B 带宽期内竞争），
    故不动推送，改攻前序。
  - 单任务相位计时显示小档（count<=16）每任务 pull~10/L1~75/L2~40µs，固定开销
    占比小；主成本仍是 B（HBM 流量，近地板）。
- 改动：
  1. `kernel.cuh`/`prologue.cuh`/`megakernel.{h,cu}`：新增 `MegaArgs::skip_prologue`。
     同一 graph 内节点路由不变（`run_aligned` 的 `nodes` 个同构节点）→ 仅首节点跑
     ROUTE_BUILD/PUBLISH_COUNTS/XBAR，其余节点跳过（数据持久、node0 的 xbar 已建立
     跨 rank 可见性），改由 `task_setup_light_body` 仅重臂每节点 flag
     （task_counter/l2_done/task_ready + 推进 cycle 计数）。`MX_NO_REUSE=1` 关闭。
  2. `bench.cu`：`run_aligned` 的 graph 捕获按节点号设 `skip_prologue`（node0=0，
     其余=2：连 xbar 一并跳过）。
  3. `compute_big.cuh`：`r24_l1_big`/`r24_l2_big` 增加 `kBPF` 模板参；tk/长尾实例
     （`kCap>0`）用 `kBPF=4`（B 预取环加深一倍），均衡大档（`kCap=0`）保持 2，
     避免 TPR=256 回退。
  4. `flags.cuh`/`kernel.cuh`：复用节点的 `mx_preclear_ct2` 跳过 xbar_barrier
     清零（该相位不执行；旧值由 epoch 单调性保证读侧安全）。
- 前 50 平均 **776.8 → 742.3~746.8µs（-4.4%，多次；含 skip-xbar 后 742.8）**。
- 回归：test/70 TPR16/64/256 = 320.7~322.9 / 599.9~606.0 / 1797.5~1804.0（基线
  348~353 / 651 / 1933，全面改善 −6~−8%）；test/71 均衡 327.4、804400=673.5、
  804300=666.6、853000=673.5（均优于基线）。
- 正确性：R=2（TPR=16/256）、R=8（TPR=64）`[总体] PASS`；`MX_NO_REUSE` 与
  复用态 L1/L2 checksum（ck1/ck2）逐位相同（804300）。

## [重要更正 261007] 不粗化（merge=1）路径结果错误，性能结论需重新基线

- **根因**：长尾场景 `merge_grid=0x3fffffff` 令 kernel 内 `merge` 保持 1（不粗化），
  物理任务数 `raw_tasks` 超过 grid 块数 → 跨波执行；多波下 wait-all/分片旗出现
  陈旧读。见 `src/compute.cuh:941` 注释（merge 粗化本为消除跨波悬挂）。
- **证据**：MXMOEMega harness `--routing-fixed-cases 1`（launch 807500）50-in-1
  CUDA graph 与 eager 对拍 `rel_max_diff=2.287e-01`；`MXK_MERGE_MAX=0` 即
  `rel=0`；`MXK_FRAG=0` 仍 2.291e-01 ⇒ 仅 merge 相关，与 frag 无关。
- **修复**：`src/bench.cu` 与 `torch/mxmegakernel/mega/__init__.py` 的 long-tail
  分支回退缺省单波粗化（merge_grid=0），保留 win_cap/frag 自适应（提交 be4f479）。
  修复后 harness case 1 `rel=0.000e+00`（1.8133ms），bench R=2/R=32 PASS。
- **重测**：`--tk-first 50` 由（错误的）744.5µs → **正确的 943.4µs**。
- **影响**：此前基线 898.5µs、以及 eb8774b/cce2fdf/2b96a0c/c0020d8/bde6343 等
  相对该基线的提升，均落在不粗化路径上，**性能结论不可比，需在正确路径上重新
  建立基线与逐项评估**。
- **后续**：若要夺回约 200µs，须修 kernel 在 merge=1 多波下的 wait-all/旗子协议
  （而非简单开启不粗化）。

## [261007] 正确路径下的重新基线（含 be4f479 merge 修复）

口径：`--tk-first 50` 平均 us/kernel；每个提交均套用「回退缺省单波粗化」修复后
独立 worktree 重建，交替两轮取均值。

| 提交 | 内容 | round1 | round2 | 均值 | 相对基线 |
|---|---|---:|---:|---:|---:|
| 49d1d34 | 基线（会话初始性能代码） | 1045.5 | 1042.9 | 1044.2 | — |
| eb8774b | 逐例自适应 win_cap | 1077.9 | 1084.0 | 1081.0 | **+3.5%（回退）** |
| cce2fdf | +中等档 33-48 走 r24 A-smem | 1037.1 | 1036.5 | 1036.8 | −0.7% |
| 2b96a0c | +kCap64 删不可达 MT8 | 1026.0 | 1023.1 | 1024.6 | −1.9% |
| c0020d8 | +kCap48 删 MT4 兜底 | 1002.6 | 1002.3 | 1002.5 | −4.0% |
| be4f479 | +bde6343 跨节点前序复用 + tk B 预取 | 944.2 | 945.5 | 944.9 | **−9.5%** |

结论：
- 正确前提下净提升 **−9.5%**（1044→945），而非早先错误路径宣称的 −17%。
- **eb8774b（逐例 win_cap）在正确路径上是回退（+3.5%）**，早先的 −4.8% 是
  不粗化路径的假象；需在正确路径上重新调 win_cap 或回退该改动。
- 有效贡献：cce2fdf(+kCap 系列合计 −4%)、bde6343 前序复用/B 预取（1002.5→944.9，
  −5.7%）。
- 进一步大收益仍取决于修复 merge=1 多波协议（约 200µs），见上一节。
## [261007] 负结果与相位分解（agent/layout/a-rwk 分支）

同会话诊断（容器 cl_test，HEAD eb335a0），使用 `-DR24_DIAG_NOALOAD`/`-DR24_DIAG_NOB`
编译消融；`test/75.fg_rwk_local_read` 为本机本地读带宽微基准。

### 1) A→RWK：正确但无收益
- 改动：L1 A pool 从 FG 对称区拆到独立 mcMalloc 普通显存（env `FP8MEGA_POOL_RWK`）。
- 正确性：`[总体] R=2 PASS`（L1 max_abs=0，L2/REDUCE 全 PASS）。
- test/70 TPR16/64/128 = 338.5/606.1/959.4 vs 基线 327.5/613.0/964.1（±3%，噪声内）；
  tk-first-30 1014.5 vs 1014.2（0.03%）。**无端到端收益，不合并**。
- 原因：A 工作集小（任务级 MB，L2 驻留）；微基准的 FG/RWK 差（~5%）只在超 L2
  且行主序散射布局出现，而 fast fragment 路径是连续读。

### 2) B：已到 HBM 峰值，空专家已跳过
- 消融：test/70 TPR64 B≈203µs（34%）、tk-first-30 B≈241µs（24%）。
  `W1+W2=288MB`，`288MB/203µs=1.42TB/s` ≈ 本机实测读峰值（test/75 contig16 1.35~1.65）。
- 空专家已由任务数公式 `ceil(count/win_cap)` 跳过（prologue.cuh:431/470）：最空
  launch(nz=19) 若读空专家需 ~8.5GB/5.7ms，远超实测，排除。
- B 成本随 `max_expert`（热点集中度）上升（≤383→~130µs；≥448→427~784µs），
  根因是**同专家多窗口的权重重读**，非空专家。布局/HBM 驻留无余量。

### 3) A 装载与相位分解（test/70 TPR64，基线 ~600µs）
- 消融：去 A≈15µs（2.5%）、去 B≈203µs、去 A+B≈217µs → 残差（MMA/epilogue/通信）
  ≈386µs（64%）。
- 相位（每 rank）：dispatch≈90、L1≈235、L2tail≈190、REDUCE(独立 kernel)≈51~105、
  mega≈544。dispatch 的 90µs 主要是 A 的跨 rank pull（远读，平台 ~10GB/s 限制）。
- 结论：B 已到峰值、A GEMM 装载被隐藏、A pull 受远读带宽限制、dispatch/REDUCE
  是跨 rank 同步/straggler。**最大且明确的剩余杠杆是修复 `merge=1` 多波下的
  wait-all/分片旗协议（约 200µs，见上一节"重要更正"）**。

### 复现
- 消融构建：`.agents/work/agent-blayout/tmp/build_bench.sh <out> [-D...]`
- 运行：`docker exec [-e ENV] cl_test bash -lc "cd <out> && ./bench ..."`
- 微基准：`test/75.fg_rwk_local_read/`

## [261007] merge=1 多波协议：GEN 代际修复 + 复现/收益（agent/layout/a-rwk 分支）

针对 perf.md「重要更正」记录的 merge=1 多波陈旧读，做了两项：
1. **实现 GEN 代际旗**（`-DNO_GEN_FIX` 可回退）：
   - `slice_flags` / `l1_prog` 旗值由 `epoch` 改为 `epoch + w/ring`
     （`pool_slot=e*ring+w%ring` 绕回时的"第几代"），wait 端用本窗同一值等待
     （compute_big.cuh `r19_pull_flags` / `r19_l1_pub_wait`）。与 l2_done 的
     单调 target 同思路（槽携带轮次身份）。
2. **env `MX_MERGE1=1`** 开通 merge=1（不粗化，多波）做 A/B。

### 实测（容器 cl_test，R=32 tk / R=2 run_scene）
- **merge=1 在 bench 上端到端正确**：
  - R=32 tk 多个不均衡 launch（807500/805400/805700/806100，含 `--win-cap 32/16`
    强制槽复用）：ck1/ck2 与 merge=0 一致（nogen 与 gen 均一致）。
  - R=2 run_scene `--pattern 1(hot)/2(mixed) --win-cap 8 --compute 1`：
    L1/L2/REDUCE-y 三行数值全 PASS、`[总体] PASS`（nogen 与 gen 均 PASS）。
- **未能在 bench 复现 perf.md 的 harness 失败**（launch 807500 rel=0.229）：
  bench 的 tk 复现 win_cap 被抬到 512 → 单窗无复用；压到 32/16 仍 PASS。
  疑 harness 几何（全模型 tokens/nslice）或 combine 跨 rank 时序不同。
- **收益巨大**（单 launch，nodes=10）：
  | 口径 | merge=0 | merge=1 | 变化 |
  |---|---:|---:|---:|
  | 807500 (win_cap=512) | 1698 | 836 | −51% |
  | 807500 win_cap=32 | 1491 | 882 | −41% |
  | 807500 win_cap=16 | 2823 | 1337 | −53% |
  | 805400 win_cap=32 | 1518 | 909 | −40% |
  | 806100 win_cap=32 | 1507 | 874 | −42% |

### 结论与风险
- merge=1 是**远超 200µs 的大机会**（不均衡例 −40~53%），且 bench 全项正确。
- 但 perf.md 记录过 harness 级失败；**合并前必须用 MXMOEMega harness
  `--routing-fixed-cases 1`（R=32、真实 y 对拍 eager）复核**。GEN 修复作为安全网。
- 复现脚本：`.agents/work/agent-blayout/tmp/{gen_repro2,gen_corr}.sh`

## [261007] merge=1 多波协议修复：harness 验证通过（agent/layout/a-rwk 分支）

GEN 代际旗修复（`compute_big.cuh`:`r19_pull_flags`/`r19_l1_pub_wait`）：
`slice_flags`/`l1_prog` 旗值 `epoch` → `epoch + w/ring`（槽代际），wait 端同式。
`torch/mxmegakernel/mega/__init__.py`：长尾分支 `merge_grid=0 → 0x3FFFFFFF`
（不粗化，多波；`MXK_MERGE1=0` 回退）。

### harness（MXMOEMega `run_mega_moe_all.sh --routing-fixed-cases N`，32 卡、
CUDA graph 50-in-1、graph vs eager 对拍）
| case | 内核 | graph-check rel_max_diff | mega_rankavg |
|---|---|---:|---:|
| 1 (807500) | 未修复(nogen) merge=1 | **3.000e-01（FAIL/报错）** | — |
| 1 (807500) | gen merge=1 | **0.000e+00 PASS** | **0.8811 ms** |
| 1 (807500) | gen merge=0（旧基线） | 0.000e+00 | 1.8164 ms |
| 17 (829200) | gen merge=1 | **0.000e+00 PASS** | 0.7582 ms |

- **bug 复现**：nogen+merge=1 rel=3.0e-1（perf.md 记 2.29e-1，同类）。
- **修复有效**：gen+merge=1 rel=0。
- **收益**：case1 **1.8164→0.8811ms（−51.5%）**；bench tk-first-50 **949.4→747.0µs（−21.3%）**，
  单 launch 最高 −53%。

### 其他本轮测过的方向（均负结果，见上节）
A→RWK 无收益；B 已 HBM 峰值；跨 rank pull 本地≈远端（非瓶颈）；`__threadfence` 可去但零收益；
fragment 必需（禁用 +38~62%）。

### 合并建议
- 需要合入：GEN 代际旗（compute_big.cuh）+ 长尾 `merge_grid=0x3FFFFFFF`
  （torch/mxmegakernel/mega/__init__.py）+ bench 的 imbalanced merge 开关。
- 实验脚手架（pool_ptr/A-RWK、PULL_*/NO_AFRAG/NO_GEN_FIX 诊断宏、MX_MERGE1 env）
  建议清理后再合。
