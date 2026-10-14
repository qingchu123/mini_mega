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

## [261009_12_50] 新基准：mybench（HEAD e8c321e，工作区含未提交的守卫/torch 改动）

口径：`scripts/run.sh mybench`，R=32，max_tokens=512，win 为 harness 默认（=max_tokens，
未设 MYBENCH_WIN）；graph nodes=20 replays=2，取 us_mean。

| 场景 | 指标 | 基准 (µs) |
|---|---|---:|
| `-s 1` 均衡 TPR=1 | us_mean | 302.75 |
| `-s 1` TPR=4 | us_mean | 320.16 |
| `-s 1` TPR=16 | us_mean | 347.95 |
| `-s 1` TPR=32 | us_mean | 468.41 |
| `-s 1` TPR=33 | us_mean | 506.25 |
| `-s 1` TPR=64 | us_mean | 644.81 |
| `-s 1` TPR=128 | us_mean | 1030.45 |
| `-s 1` TPR=256 | us_mean | 1909.77 |
| `-s 1` TPR=512 | us_mean | 3698.75 |
| `-s 2 -top 50` | 50 launch 平均 | **3274.10** |

> 注：`-s 2 -top 50` 的 50 个 launch 均为 win=512（harness 默认），未启用自适应 win。
> 首次 `-s 1` 因 GPU 队列创建超时（环境问题，ioctl 重试约 7 分钟）被 timeout 终止，重跑通过。

## [261009_13_13] 固定逻辑 slice 协议（去除 kBarNsliceEffIdx 广播）

口径同 [261009_12_50]。HEAD 基准 e8c321e。

| 场景 | 指标 | 基准 [261009_12_50] (µs) | 本次 (µs) | 变化 |
|---|---|---:|---:|---:|
| `-s 1` TPR=1 | us_mean | 302.75 | 306.15 | +1.1% |
| `-s 1` TPR=4 | us_mean | 320.16 | 317.80 | -0.7% |
| `-s 1` TPR=16 | us_mean | 347.95 | 346.91 | -0.3% |
| `-s 1` TPR=32 | us_mean | 468.41 | 473.77 | +1.1% |
| `-s 1` TPR=33 | us_mean | 506.25 | 506.12 | 0.0% |
| `-s 1` TPR=64 | us_mean | 644.81 | 647.92 | +0.5% |
| `-s 1` TPR=128 | us_mean | 1030.45 | 1038.20 | +0.8% |
| `-s 1` TPR=256 | us_mean | 1909.77 | 1913.66 | +0.2% |
| `-s 1` TPR=512 | us_mean | 3698.75 | 3708.25 | +0.3% |
| `-s 2 -top 50` | 50 launch 平均 | 3274.10 | **3149.45**（复测 3149.33 / 3148.75） | -3.8% |

> 注：`-s 1` 各档在 ±1.2% 内。`-s 2 -top 50` 下降的原因未确认：新旧列段写法数据量等价，待查，暂不计为已确认优化。

## [261009_19_40] 结构整理（rwk/unk 结构体化 + torch 自有分配）— 无性能变化记录

口径：`scripts/run.sh mybench -s 1 -tprs 16`（同机，μs，us_mean）。本轮为结构性改动，
不计为优化；列出是为了证明没有可归因的回退。

| 版本（提交） | TPR=16 us_mean | 说明 |
|---|---:|---|
| `09e5035`（删除 bench 后） | 348.10 | 结构整理前基线 |
| `1528240` | 351.36 | harness 整块分配 |
| `9b1be8c`（2b-A 后） | 349.64 | kernel 读取改为 rwk/unk |
| `e2728ff`（2b-B 后） | 341.57 | peer 表统一为 unk_peers |

结论：四次采样均落在 341–352 µs，与既有波动（约 ±1–2%）一致，无可归因回退。
正确性：每步均通过 check smoke 5/5、子集 12/12、top50 52/52。

torch 路径（cl_test，32 GPU）：未做性能测量，仅验证编译、建档、预热、destroy 无非法访存。

## 小档优化：TPR 17–32（commit `89db243`）

> 口径：`test/mybench`（R=32, nodes=20, replays≥2, **win=128**）；**不等同**上方 `./bench` 口径，故单独列出。

| TPR（均衡） | 改前 `paired<2,2,2,4>` | 改后 `r19_l1_big<4,2,4>` | 提升 |
|---:|---:|---:|---:|
| 17 | 707 | 656 | **+7.2%** |
| 20 | 720 | 654 | **+9.1%** |
| 24 | 721 | 673 | **+6.7%** |
| 28 | 717 | 665 | **+7.3%** |
| 32 | 711 | 674 | **+5.2%** |

- top50（TK 实录, win=128, 50 launch 的 `us_mean` 求和）：48878 → **48096**（−1.6%）。
- 正确性：`check --smoke` 8/8 PASS（新增 balanced R=24/32 覆盖 17–32 档）。
- 机制：`count∈(16,32]` 由 MT2 的 `paired<2,2,2,4>` 改为 MT4 的 `r19_l1_big<4,2,4>`，
  B 在 4 个 M-tile 间共享（权重读遍数减半）。

## 小档优化：≤16 热路径（commit `929a9c7`）

> 口径：`test/mybench`（R=32, nodes=20, replays≥2, win=128）。

| TPR（均衡） | 改前 | 改后 | 提升 |
|---:|---:|---:|---:|
| 1 | 323 | 312 | **+3.4%** |
| 4 | 336 | 324 | **+3.6%** |
| 8 | 344 | 334 | **+2.9%** |
| 16 | 358 | 355 | **+0.8%** |

- 机制：`r19_small_task` 的 g_l1=2 回退由 `paired<4,kNT,1,2>`（PF4）改 `r19<1,2,2>`，
  small task 不再实例化 paired → 省 PF4 预取环寄存器。
- top50（win=128 求和）：48096 → **47912**（−0.4%）。

## 当前基线快照（commit `40136dc`）

> 口径：`test/mybench`（R=32, nodes=20, replays≥2）；TPR = 每 rank token（均衡）。

| TPR | 均衡(默认选窗) | 均衡(win=128) |
|---:|---:|---:|
| 1 | 318 | 313 |
| 4 | 328 | 326 |
| 16 | 362 | 359 |
| 32 | 669 | 667 |
| 33 | 508 | 497 |
| 64 | 648 | 644 |
| 128 | 1009 | 1008 |
| 256 | 1871 | 1787 |
| 512 | 3632 | 3418 |

- **top50**（TK 实录, win=128）：sum=**47497µs**，mean=**950µs**/launch。
- L1 现状：大档 `r24<8,8,2>`；17–32 与 ≤16 用 `r19`（`89db243`/27f76f1）；全档 fragment（`27f76f1`）；生产 dispatch 无 paired（`e16637e`）。
- 内核资源（`MT_DUMP_RES=1`）：**regs=256（顶格）、local≈2.5KB**（存在寄存器溢出；定位脚本见 `40136dc`）。降 acc 档位/预取对 regs/local 基本无影响；`__noinline__` 拆分反而增大 local。

## [261010_2200] L1/L2 拆分 kernel（A/B 端到端）— 净回退，默认关闭

> 动机：融合任务体 local≈2.5KB 溢出，猜想拆成「L1 kernel + L2 kernel」可让两段各自减负。
> 实现：`MegaPhase{Both,L1,L2}` 模板，`MX_L1L2_SPLIT=1` 运行期切换（默认 0=原融合路径）。
> L1 kernel = 完整前序(route/publish/xbar)+DISPATCH+L1；L2 kernel = 轻量前序(不 route/xbar/
> task_setup)+静态任务划分+L2+COMBINE。每 round `current_time[0]`/`[1]` 各 +1（仅 L1 前序推进），
> 数据槽改每窗口独占 `e*slot_win+w`（`slot_win=ceil(kMaxTPR/win_cap)`，缓冲复用现结构不加尺寸）。

**正确性**（`check --smoke`，R=32）：两种模式均 **8/8 PASS**（eager + graph×1 + graph×50）。

**性能 A/B**（`test/mybench`，R=32，nodes=20）：

| 场景 | 融合基线 us_mean | 拆分 us_mean | 变化 |
|---|---:|---:|---:|
| 均衡 TPR=16 | 375.6 | 410.1 | +9.2% |
| 均衡 TPR=32 | 669.7 | 725.5 | +8.3% |
| 均衡 TPR=64 | 655.7 | 703.5 | +7.3% |
| 均衡 TPR=128 | 1017.3 | 1094.8 | +7.6% |
| 均衡 TPR=256 | 1888.6 | 2044.5 | +8.3% |
| 均衡 TPR=512 | 3683.5 | 3932.2 | +6.8% |
| TK top50 sum（2 次） | 113502 / 113524 | 116821 | +2.9% |

**内核资源**（`MT_DUMP_RES=1`）：融合 `full` local=2544B；`L1full` local=1680B；`L2full`
local=1984B —— 拆分确实降低了单核 local/栈，但两者仍 **regs=256** 且仍有溢出。

**结论**：**拆分是净回退（均衡 +7~9%，TK +2.9%），已否决**。根因：融合的寄存器占用取 L1/L2
较大者而非之和，local 下降不足以抵消「阶段间全局栅栏 → 失去窗口间 L1/L2 重叠 + 尾部等待」的损失。
保留 `MX_L1L2_SPLIT` 仅作实验开关（默认关，零开销）。后续若再试，应先解决融合体溢出本身
（如对齐 acc 生命周期），而非拆 kernel。

## [261011_12_43] L1 中档调优（B/D）— 17..32 与 49..63

> 动机：L1 统一实验（`docs/14~16`，分支 `agent/l1sweep/uniform`）发现现状两处中档次优。
> 口径：`test/mybench -R 32 -s 1 -nodes 3 -replays 1`（us_mean，跨 rank）。完整矩阵见 docs/16。

改动（`src/compute_big.cuh`，`r19_big_task` L1 段）：

1. **B**：17..32 档 `r19_l1_big<4, R19_L1_PF, 4>` → `r19_l1_big<2, 1, 4>`
   （同 1 组权重、acc 128→64、预取环 2→1）。
2. **D**：49..64 档外层去掉 `count % (4*kM)==0`；整块(64)无掩码、**49..63 尾块掩码**
   `r24_l1_big<4, R24_KS, R19_L1_BATCH, true, 4, BPF>`（原 49..63 掉进 r19 慢回退）。

**A/B**（同网格，改动前 = 基线）：

| TPR | 基线 us | B/D us | 提升 |
|---:|---:|---:|---:|
| 18 | 606 | 462 | **+24%** |
| 24 | 630 | 490 | **+22%** |
| 30 | 626 | 488 | **+22%** |
| 54 | 826 | 648 | **+22%** |
| 60 | 861 | 663 | **+23%** |
| 36 / 42 / 48 | 507 / 513 / 495 | 503 / 526 / 511 | ±3%（噪声） |
| 66 / 90 / 128 / 256 | 724 / 957 / 981 / 1846 | 725 / 951 / 976 / 1853 | ~0 |

**正确性**：`check --smoke` **PASS**（8 用例，含 TPR=16/24/32/128 与 graph×50；D 的 49..63 掩码路径覆盖）。

**结论**：中段两处 −22~24%，其余无回退。大 count（>64）保持 r24 MT8 async 两段式不变。

## [261011_2018] 寄存器 spill 治理：大档 L1/L2 选择性 noinline 隔离 — 内核 local −17%，性能中性偏升

> 背景：生产核 `mega_kernel<true>` 每线程 **regs=256（顶格）+ local≈3.0KB**，溢出真实存在
> （见 `reg_spill.md`）。根因结构性：核心 GEMM 体的 `acc`（MT8/B2 达 **128 floats**）+
> A/B 预取环 + forceinline 的 L1/L2 body 叠加，超出 256 寄存器预算。
> 禁用（`-DR19_L1BIG_ISO/__forceinline__`）则 r19_big_task 的 acc 峰值传染进任务体。

改动（`src/common.cuh`、`src/compute_big.cuh`、`src/compute.cuh`）：

1. **E1s·L1**：`r19_big_task` 内 **kMT>=2** 的 L1 档改经 `r19_l1_big_iso`（`__noinline__` shim，
   `R19_L1BIG_ISO`）隔离；small task 的 kMT=1 热路径保持内联。
2. **L2**：`r19_big_task` 内中/小档 L2 body 改经 `l2_gemm_body_iso`（`__noinline__` shim，
   `R19_L2BODY_ISO`）隔离（acc 单批最多 96 floats）；small task 调用点保持内联。

**内核资源**（`MT_DUMP_RES=1`，`-s 1 -tprs 16`）：

| 版本 | regs | local（栈帧/溢出） |
|---|---:|---:|
| 基线 | 256 | **3096 B** |
| 仅 L1 隔离（E1s） | 256 | 2856 B |
| **L1+L2 隔离（本改）** | 256 | **2568 B（−17%）** |

静态反汇编（`mxobjdump`+`frame.py`）：`r19_big_task` 栈帧 **1344 → 528 B**（−61%，
栈 ld/st 905→497）；`r24_l2_big<8,8,2,true,8,2>` 640 B（noinline，内部 acc+推送）。

**A/B**（`mybench -R 32 -s 1 -tprs 8,16,32,64,128,256 -nodes 3 -replays 2`，与基线交错 3 次均值）：

| TPR | 基线 us | L1+L2 iso us | 变化 |
|---:|---:|---:|---:|
| 8 | 342.0 | 342.9 | +0.3% |
| 16 | 361.4 | 358.1 | −0.9% |
| 32 | 517.8 | 482.8 | **−6.8%** |
| 64 | 630.2 | 622.2 | −1.3% |
| 128 | 959.0 | 950.5 | −0.9% |
| 256 | 1692.5 | 1668.7 | −1.4% |

**-s2 top50**（TK 实录，交错 2 次）：基线 mean **792.9µs** → **771.9µs（−2.6%）**。

**正确性**：`check --smoke` **PASS**（8 用例，含 TPR=32 graph×50）。

**结论**：spill 显著下降（内核 local −17%、最重的任务体 −61%），且性能**中性偏升**
（中档 −6.8%、TK −2.6%，无档回退 >2%）。溢出的剩余部分在 `r24_l2_big`（noinline 自包含，
acc=128 结构性），强行再降（降 acc/拆 kernel）已证净回退（`reg_spill.md` §10）。
可用 `-DR19_L1BIG_ISO=__forceinline__` / `-DR19_L2BODY_ISO=__forceinline__` 回退。

## [261012_03_04] `-s2` 细粒度派发（NS=8 / 跨 block K 切）— 净回退，默认关闭

> 动机：`R19_BLOCK_LPT`（NS=4）因任务量化无法压低最忙 block 上界（见 README
> 「块级代价感知 LPT」）。尝试两种更细粒度派发，目标 `-s2 top50` 765→~400µs。
> 改动仅 `src/compute.cuh` / `src/compute_big.cuh`；开关默认关（关时逐字等价基线）。
> 口径：`test/mybench -R 32`；s1 = `-s1 tprs 32,128,256 -nodes 3 -replays 2`，
> s2 = `-s2 top50 -nodes 20 -replays 3`，us_mean（跨 rank）。

**Stage 1：N 切 NS=8**（`-DKNSLICE_OVERRIDE=8`，±`-DR19_BLOCK_LPT=1`）

| 指标 | 基线 NS=4 | NS=8 | NS=8+LPT |
|---|---:|---:|---:|
| s1 TPR32 | 485.4 | 743.6 | 762.9 |
| s1 TPR128 | 946.1 | 1208.8 | 1225.0 |
| s1 TPR256 | 1678.6 | 2239.6 | 2277.8 |
| s2 top50 mean | 767.9 | 1086.8 | — |

→ s1 +28~53%、s2 +42%，**否决**（N 切使 A 装载 ×N + 协议任务 ×2）。

**Stage 2：跨 block K 切**（`-DR19_BLOCK_KSPLIT=1`，`win_cap≤64` 启用）

任务 = `(e,w,slice,kseg)`，`raw_tasks=s_cum[kEPR]*kNSlice*kNK`；owner(kseg0) 跑
pull/staging/分片旗/L2/tail，其余段只做 K 部分和（FP32 atomicAdd → `l1_partial`，
`l1_part_cnt` 判最后段 → `r19_l1_reduce` → 发 `l1_prog`）。

| `kNK` | s2 top50 mean | s1 TPR32 | s1 TPR128 | s1 TPR256 |
|---:|---:|---:|---:|---:|
| 1（基线） | 767.9 | 485.4 | 946.1 | 1678.6 |
| 2 | 1003.6 | — | — | — |
| 4 | 1419.1 | — | — | — |
| 8 | 1970.2 | 1246.8 | 979.1 | 1690.5 |

块级 L1（804900 / rank28）：基线满窗块 ~383k–416k；K 切(kNK=8) min 212k /
max 472k（ratio **2.22×**，未收敛 ≲1.2×）。

→ K 切随段数单调变差，**否决**。根因：①每窗 NS×NK 个任务把窗口独占整张 grid，
由基线「8 窗并行」退化为「1 窗串行」，窗内 pull→段→reduce→L2 的串行段无法被其它
窗掩盖（kNK 越小回退越小）；②FP32 部分和 atomicAdd 流量在小窗下不可忽略；
③段代价仍 ∝ 行数 + FCFS 按窗连发，块级未收敛。

**正确性**：关 `check --all` **762/762**（79s）；开 `check --all` **762/762**（119s）；
`check --smoke` 开/关均 8/8 PASS（<10s）。**未接近 400µs**；两开关保持默认关。

## [261012_08_23] 选窗重标定（ring-aware + 倾斜门控）— `-s2 top50` 765.7→750.9µs（−1.9%）

> 动机：诊断（docs/21）显示 `-s2` 关键路径 = 最忙 block 的 L1+L2，窗口容量 `win`
> 直接影响；docs/11 遗留「逐 case oracle 901 vs 规则 942 无法廉价预测」。本任务聚焦
> **每 case 的 win_cap 选择**，仅改 `src/win_select.cuh`（`select_win` 重标定）。
> 口径：`test/mybench -R 32`；s1 = `-s1 tprs 32,128,256 -nodes 3 -replays 2`，
> s2 = `-s2 top50 -nodes 20 -replays 3`，us_mean（跨 rank），各取 3 次均值。

### 1. 逐 case win 扫描（`MT_FORCE_WIN∈{32,48,64,96,128}`，各 3 次均值）

- 全量 mean：固定 32=971.4 / **48=756.6** / 64=770.9 / 96=807.3 / 128=853.6。
  **最佳固定 = 48**；但逐 case 最优（oracle）= **729.7**（−5.0%），选中分布
  `{48:18, 64:13, 96:14, 128:5}`——**没有任何单一固定 win 接近 oracle**。
- 逐 case 最优与 `max_exp` 弱相关：既有多例 `max_exp≈470` 偏好 96/128，也有
  `max_exp≈470` 偏好 48；线性组合 `W(win)`/`max_exp`/窗数等特征拟合均退化为
  固定 48（见下「无法廉价预测」）。当前代价模型（waves·t_task）选中 `{48:39,
  64:4, 128:7}`，其中 7 例 128 **全部为负收益**（如 804500 +12.7%、807700 +16.1%）。

| case | max_exp | 当前win | 当前us | 最优win | 最优us | 收益 |
|---|---:|---:|---:|---:|---:|---:|
| 804300 | 362 | 48 | 739.2 | 96 | 699.5 | +5.4% |
| 804400 | 176 | 48 | 711.0 | 96 | 710.5 | +0.1% |
| 804500 | 292 | 128 | 835.6 | 48 | 729.4 | +12.7% |
| 804600 | 327 | 48 | 738.0 | 48 | 740.1 | −0.3% |
| 804700 | 265 | 64 | 661.3 | 64 | 653.6 | +1.2% |
| 804800 | 425 | 128 | 840.3 | 96 | 712.4 | +15.2% |
| 804900 | 190 | 48 | 776.6 | 48 | 777.3 | −0.1% |
| 805000 | 473 | 48 | 739.8 | 48 | 746.2 | −0.9% |
| 805100 | 365 | 48 | 720.1 | 48 | 717.4 | +0.4% |
| 805200 | 390 | 128 | 851.2 | 128 | 850.4 | +0.1% |
| 805300 | 220 | 48 | 692.5 | 48 | 687.6 | +0.7% |
| 805400 | 406 | 48 | 975.0 | 128 | 818.9 | +16.0% |
| 805500 | 474 | 48 | 687.8 | 64 | 671.9 | +2.3% |
| 805600 | 191 | 48 | 754.9 | 48 | 739.2 | +2.1% |
| 805700 | 495 | 48 | 862.7 | 128 | 812.8 | +5.8% |
| 805800 | 479 | 48 | 753.2 | 96 | 704.2 | +6.5% |
| 806000 | 254 | 48 | 732.4 | 48 | 734.3 | −0.3% |
| 806100 | 478 | 48 | 798.6 | 96 | 767.4 | +3.9% |
| 806200 | 505 | 48 | 921.7 | 64 | 844.8 | +8.3% |
| 806400 | 382 | 48 | 776.3 | 96 | 710.3 | +8.5% |
| 806500 | 186 | 48 | 761.9 | 48 | 749.9 | +1.6% |
| 806700 | 384 | 48 | 685.7 | 64 | 659.3 | +3.9% |
| 806800 | 355 | 48 | 767.4 | 64 | 720.7 | +6.1% |
| 806900 | 405 | 64 | 680.2 | 64 | 684.1 | −0.6% |
| 807000 | 168 | 48 | 766.2 | 96 | 711.8 | +7.1% |
| 807100 | 395 | 48 | 712.0 | 64 | 686.5 | +3.6% |
| 807200 | 197 | 48 | 742.0 | 64 | 734.0 | +1.1% |
| 807400 | 189 | 48 | 737.8 | 48 | 749.2 | −1.5% |
| 807500 | 477 | 48 | 827.4 | 96 | 784.9 | +5.1% |
| 807600 | 201 | 48 | 714.5 | 48 | 716.2 | −0.2% |
| 807700 | 272 | 128 | 799.6 | 64 | 670.5 | +16.1% |
| 807800 | 474 | 48 | 816.2 | 128 | 796.6 | +2.4% |
| 807900 | 300 | 64 | 776.9 | 96 | 712.1 | +8.3% |
| 808000 | 338 | 64 | 853.6 | 64 | 831.5 | +2.6% |
| 808100 | 370 | 48 | 708.4 | 96 | 693.3 | +2.1% |
| 808200 | 249 | 48 | 634.1 | 64 | 615.0 | +3.0% |
| 808300 | 215 | 48 | 736.0 | 48 | 733.3 | +0.4% |
| 808400 | 416 | 128 | 844.4 | 48 | 734.4 | +13.0% |
| 808500 | 325 | 48 | 680.6 | 48 | 690.4 | −1.4% |
| 808600 | 267 | 48 | 717.7 | 96 | 696.5 | +2.9% |
| 808700 | 425 | 48 | 768.0 | 96 | 703.7 | +8.4% |
| 808800 | 174 | 48 | 769.3 | 48 | 768.4 | +0.1% |
| 808900 | 463 | 48 | 769.9 | 48 | 766.1 | +0.5% |
| 809000 | 337 | 128 | 795.0 | 48 | 698.1 | +12.2% |
| 809200 | 233 | 48 | 713.7 | 64 | 704.9 | +1.2% |
| 809300 | 374 | 48 | 768.5 | 96 | 695.7 | +9.5% |
| 809400 | 460 | 128 | 798.1 | 48 | 731.7 | +8.3% |
| 809500 | 181 | 48 | 714.1 | 64 | 712.2 | +0.3% |
| 809600 | 469 | 48 | 957.3 | 128 | 808.7 | +15.5% |
| 809700 | 481 | 48 | 810.9 | 96 | 696.7 | +14.1% |
| **均值** | | | **767.9** | | **729.7** | **+5.0%** |

### 2. 新 `select_win` 规则（`src/win_select.cuh`）

根因：`win_cost` 用「总窗数 waves · 每窗固定成本 t_task(win)」建模，对**均衡**成立
（总任务量决定，均衡大档最优 128）；但**倾斜**负载关键路径是最忙专家，且
`compute.cuh` 的 L2 ring（`kRing=8`）使 `ceil(c_e/win)>8` 的窗在 `l2_gate_wait`
上串行。实测倾斜时 128 最差（小专家多、宽窗权重/缓冲浪费）。故按分布形态分治：

1. **倾斜判定** `max_e c_e · E > 3 · Σc_e`（max/mean>3；均衡=1，top50≈10~32）；
2. **倾斜**：win 下限抬到 `ceil(max_exp / kRing)`（消 ring 串行台阶），上界压到 96
   （去 128）；`ceil(max_exp/kRing)>96` 时说明 96 仍不足，放开到 `cap_hint`；
3. **非倾斜**：原代价模型（含 128，均衡大档实测最优）；
4. 在 `[floor, cap]` 内取 `W(win)·t_task` 最小候选。

即 top50（全倾斜）实际只选 `{48,64}`：`win=48` 当 `ceil(max_exp/48)≤8`（max_exp≤384），
否则 64。扫描预测均值 **751.2**（−1.9% vs 当前 765.5，−0.7% vs 最佳固定 48）。

> **安全回退**：`MXK_WIN_LEGACY=1` 走旧代价模型（`mxk::select_win_legacy`），默认关。
> 实测：新规则 751.8µs / legacy 764.6µs（同次单跑），与上表一致。

> **无法廉价预测 oracle**：对 `(W(win), ceil(max_exp/win), ring 超窗数, ngt, 活动专家数)`
> 等特征做线性拟合/阈值搜索，argmin 均退化为固定 48；oracle 的 5% 空间来自
> 逐 case 关键路径（与具体 rank/块路由相关），`c_e` 聚合量不可分。故只回收部分。

### 3. A/B（各 3 次均值，us_mean）

| 场景 | 改前（旧模型） | 改后（新规则） | 变化 |
|---|---:|---:|---:|
| **`-s2 top50` mean** | **765.7** | **750.9** | **−1.9%** |
| `-s2` 逐次 | 767.9 / 764.8 / 764.4 | 752.0 / 750.3 / 750.3 | — |
| `-s2` win 分布 | {48:39, 64:4, 128:7} | {48:26, 64:24} | win_global 128→64 |
| s1 TPR=32 | 485.4 | 488.8 | +0.7% |
| s1 TPR=128 | 946.1 | 953.7 | +0.8% |
| s1 TPR=256 | 1678.6 | 1675.7 | −0.2% |

s1 三档均在 ±1%（均衡走非倾斜分支，与旧模型逐字等价，差异为运行噪声）；
s2 的 win_global 由 128 降到 64，额外收紧 ffn/l2 缓冲。

### 4. 门禁

- `check --smoke`：**PASS 8/8**，7.46s（<10s）。
- `check --all`：**PASS 762/762**，77.8s（`timeout 400` 内）。
- s1 回退 <1%（远小于 5% 阈值），不触发回退。

### 5. 结论

选窗从「旧模型 765.7」降到「新规则 750.9」（−1.9%），优于最佳固定 win=48（756.6）。
收益来自：① 去掉旧模型在倾斜负载误选的 128（7 例全负收益）；② ring 下限避免
最忙专家的串行台阶。距逐 case oracle（729.7）仍差 2.9%，该空间需关键路径级信息，
非 `c_e` 聚合量可廉价预测。**距终极 400µs 仍远**，本任务仅覆盖 win 维度。

## [261012_09_40] 方向1：B（权重）smem 异步双缓冲（`R24_BSMEM_B`，默认关）— 中性偏负

> 动机：诊断（同会话消融）显示 `-s2 top50` 中 **B 装载裸露 ≈211µs（28%）**
> （`R24_DIAG_NOB` 755.5→544.1，nodes=3 replays=1），L1≈352µs/L2≈172µs，
> 聚合 MMA 利用率仅 ~2.5%（延迟/同步受限）。目标：把 B 也走 smem 异步双缓冲以
> 隐藏其装载延迟，向 400µs 靠近。
> 口径：`test/mybench -R 32`；s1 = `-s1 tprs 32,128,256 -nodes 3 -replays 2`，
> s2 = `-s2 top50 -replays 3`，us_mean（跨 rank）。改动仅 `src/compute_big.cuh`、
> `src/kernel.cuh`。开关默认关（关时逐字等价基线）。

### 1. 实现（`src/compute_big.cuh`）

- `r24_l1_big` / `r24_l2_big` 增加编译期模板参 `int kBsmB=0`（环深）。`kBsmB>0` 时：
  - 新 `r24_issue_b<kBatch>`（每 kstep 每线程发 `kBatch` 个 16B `__builtin_mxc_ldg_b128_bsm`，
    源地址与 `r19_load_b16` 逐字一致）；`r24_lds_b<kBatch>` 从 smem LDS.128 读 B。
  - B 环 smem 紧接 A 双缓冲之后（实例偏移 `2*kMT*kKS*512`）；`kABSharedBytes` 默认
    保持 A 双缓冲 64KB 不变、开启时放大到 112KB（med L1 `<3,16,4>` = 48KB A+64KB B）。
  - 分档：仅 `win_cap<=64`（tk/长尾）的 r24 实例传 `kBsmB=R24_BRING(=2)`；均衡大档
    （`count>64`，MT8）保持直装。
- **关键坑（硬件队列）**：`ldg_b128_bsm` 的拷贝计入 **gvmcnt**，**不是 bsmcnt**。
  `arrive(4096)=bsmcnt(0)` 等不到 B → 早期版本出现非确定性 NaN/错值（`check` 间歇 FAIL）。
  正解：`R24_WAIT_BSM_N(N) = __builtin_mxc_arrive(64 + N) = gvmcnt(N)`；标准段允许
  「环深-1」个 kstep 的 B 在飞、尾部 `gvmcnt(0)` 全等。A 的可见性仍由 stage 顶
  `bsmcnt(0)+__syncthreads()` 保证。（`mxobjdump` 反汇编确认 `arrive` 的计数器选择位。）

### 2. 正确性

- 默认关：`check --all` **762/762 PASS**（76.7s）、`check --smoke` **8/8 PASS**（<10s），
  s1/s2 与基线逐字一致。
- 开启（`-DR24_BSMEM_B=1`）：`check --all` **762/762 PASS**（77.8s）、`check --smoke`
  **8/8 PASS**（<10s，多次复跑稳定）。

### 3. 性能（同会话基线 s2=749.5）

| 场景 | 默认（直装 B） | B-smem 环2 | 变化 |
|---|---:|---:|---:|
| **s2 top50 mean** | **749.5** | **763.1** | **+1.8%** |
| s1 TPR32 | 482.1 | 483.5 | +0.3% |
| s1 TPR128 | 947.4 | 967.9 | +2.2% |
| s1 TPR256 | 1686.2 | 1681.0 | −0.3% |

`s2` 微回退 >5% 阈值内；环深 1（全等，无预取）= 864µs（更差，证明预取有效但被开销抵消）。

### 4. B 延迟隐藏诊断（`R24_DIAG_NOB`，nodes=3 replays=1）

| 口径 | base | NOB | B 暴露 |
|---|---:|---:|---:|
| 默认（直装 B） | 755.5 | 544.1 | **211.4µs** |
| B-smem 环2 | 769.9 | 677.3 | **92.6µs** |

→ B-smem 确实把「B 延迟暴露」砍掉 **~119µs**，但 **NOB 地板抬升 133µs**
（544→677）——B 经 smem 多一遍「bsm 写 + LDS 读」，占用 smem 端口带宽；且 smem
从 66KB 增到 115KB、L1 carveout 缩小。净结果为负。

### 5. 结论

**B 是带宽受限（≈191µs @1.4TB/s）而非可隐藏延迟**：smem 双缓冲隐藏了延迟，却
以等量的 smem 端口流量为代价，净中性偏负。要压低 B 需**减少权重字节/提高复用**
（窗口/布局类），而非掩盖延迟。**未接近 400µs**；开关默认关
（`R24_BSMEM_B=0` 编译期消除，关时 `check --all` 762/762、s1/s2 逐字等价基线）。

## [261012_11_10] 方向3：通信/REDUCE 与计算重叠 — 净中性/微回退，默认关（否决策略，附关键结论）

> 目标：`-s2 top50` ~751µs 尽量下降（终极 400µs）。本任务聚焦「把通信/REDUCE 与
> 计算重叠」。改动：`src/combine.cuh`（`reduce_body` 拆出 `reduce_tile_body` + wrapper；
> 新增 `MXK_FUSE_REDUCE` 开关）、`src/kernel.cuh`（mega 尾融合 REDUCE）、
> `src/megakernel.cu`（`launch_full` 融合分支 + `to_mega_args` 传 y）、
> `src/compute.cuh`（`MegaArgs::y` 字段）。开关默认关（`MXK_FUSE_REDUCE=0` 编译期消除，
> 关时逐字等价基线）。
> 口径：`test/mybench -R 32`；s1 = `-s1 tprs 32,128,256 -nodes 3 -replays 2`，
> s2 = `-s2 top50 -replays 3`，us_mean（跨 rank）。

### 1. 关键诊断（本机，R=32）

先把「通信地板」逐项消融量化（`-s2 top50` us_mean；`CM_SKIP_GEMM` 见
`src/compute.cuh:99`，`PULL_LOCAL_DIAG` 见 `src/compute_big.cuh:1490`，
`MX_SKIP_REDUCE` 见 `src/megakernel.cu`）：

| 口径 | s2 top50 mean | 差值（=该分量在关键路径上的边际成本） |
|---|---:|---:|
| 基线 | 753.3 | — |
| `CM_SKIP_GEMM=1`（纯通信+REDUCE） | **174.9** | 地板 174.9（非与 GEMM 相加项） |
| `MX_SKIP_REDUCE=1` | 739.0 | **REDUCE ≈ 14.3µs** |
| `PULL_LOCAL_DIAG=1`（A 远端读→本地读） | 740.4 | **A 远读 ≈ 13.0µs** |
| `R30_DIAG_NOPUSH=1`（去 COMBINE 融合推送） | 805.7 | 反而更慢（推送已重叠，非暴露项） |

s1（同法，`PULL_LOCAL_DIAG`）：

| 档 | 基线 us | PULL_LOCAL us | A 远读 |
|---|---:|---:|---:|
| TPR=128 | 949.4 | 884.9 | **≈ 64.5µs** |
| TPR=256 | 1670.2 | 1625.4 | **≈ 44.8µs** |
| TPR=256 `MX_SKIP_REDUCE` | 1670.2 | 1637.6 | REDUCE ≈ 32.6µs |

**核心结论**：`CM_SKIP_GEMM` 的 174.9µs「地板」**不等于**可回收量——在完整
路径中，A pull / COMBINE 推送 / 跨 block 分片 wait-all 大部分已被 GEMM 计算掩盖
（PULL 边际仅 13µs、NOPUSH 反而更慢）。**通信真正暴露在 s2 关键路径上的边际
只有 REDUCE 14µs + A 远读 13µs ≈ 27µs（3.6%）**；s1 主要暴露项是 A 远读
45~65µs。故方向3 对 s2 的理论上限约 749→722µs，**远达不到 400µs**。

### 2. 候选1（REDUCE 并入 mega 尾，`MXK_FUSE_REDUCE=1`）

实现：`mega_kernel` 任务循环后各 block 就地按 `item=(token,half)` 轮分执行
`reduce_tile_body`（`src/kernel.cuh`）；`launch_full` 不再单发
`mega_reduce_kernel`。epoch/cyc 取自本轮 task_setup 已自增的
`current_time[0][1]`，与独立核一致；y 由 host 经新增 `MegaArgs::y` 传入。

**正确性**：开 `check --smoke` **8/8 PASS**（<10s，含 graph×50）；关
`check --all` **762/762 PASS**（78.6s）。

**A/B**：

| 场景 | 基线（默认关） | 融合 | 变化 |
|---|---:|---:|---:|
| s2 top50 mean | 749.1 | 755.0 | +0.8% |
| s1 TPR=32 | 485.2 | 492.8 | +1.6% |
| s1 TPR=128 | 950.5 | 983.8 | **+3.5%** |
| s1 TPR=256 | 1675.5 | 1717.1 | **+2.5%** |

→ **净中性/微回退，默认关**。根因（实测）：
1. **SM 已被 mega 占满**：`MT_DUMP_RES=1` 显示 `full regs=256 smem=66160`，
   512 线程 ×256 regs 已吃满单 SM 寄存器预算，reduce 无法与 GEMM 的其它 block
   真正并发（融合只是把 reduce 排到各 block 任务之后，无重叠窗口）。
2. 独立 REDUCE 用 `2*tokens` 个 block；融合后仅 `kGrid=32` 个 block，s1 大档
   （tokens=256 → 需 512 block）并行度降 16×，故 s1 明显回退（s2 tokens=16，
   2*tokens=32=grid，故中性）。

### 3. 候选2/3 评估（未产生正收益）

- **候选2（pull 提高在飞请求数/预取）**：pull 热路径已是最优——`win=48/64` 时
  `u4_slice=96` 走 `[H2]` 三条远端装载全发再连续 store（`src/compute_big.cuh:1503`），
  MLP 已满；且 A 是 FG 远读带宽受限（平台 ~10GB/s，perf.md 前记）。s2 边际仅 13µs。
- **候选3（COMBINE 推送与计算重叠）**：R29 融合推送已生效；`R30_DIAG_NOPUSH`
  去推送反而更慢（805.7µs），说明推送已完全隐藏在 L2 内，无剩余串行尾。
- **候选1 的跨 launch 变体（独立 stream）**：本机验证 MACA 流捕获语义——`BeginCapture
  (ThreadLocal)` 下若侧流未在 `EndCapture` 前 join，返回 `ErrInvalidState(904)` 且
  无图产出（`.agents/work/impl1/tmp/cap_test2.cu`）。bench/check 逐节点调用
  `launch_full` 且不提供 finalize 钩子，**N 节点必留 ≥1 个未 join 的 REDUCE 侧流工作**，
  跨节点重叠在现有 harness 下不可实现（除非改 test/mybench.cu/check.cu 的捕获模式，
  属越界）。

### 4. 结论与遗留

- **方向3 对 s2 的边际上限 ≈ 27µs（REDUCE 14 + A 远读 13），已基本被现有实现
  吃掉**；融合 REDUCE 净中性/微回退，默认关。**未接近 400µs**。
- 通往 400µs 的**剩余杠杆不在通信/重叠**，而在 GEMM 本体（s2 中 ≈576µs）与
  跨 block 分片 wait-all 的**任务量化**（每窗 NS=4 片、鸽笼上界，见 README
  「块级代价感知 LPT」）。
- 遗留：若允许改 harness（Global 捕获 + 显式 finalize），可尝试跨节点
  side-stream REDUCE；但本机 SM 已满，预期仍受并发窗口限制。

## [261012_10_01] 方向2：17..32 档 L1 预取 PF1→PF2（每窗延迟隐藏）— s2 −2.1%、s1 TPR32 −4.0%

> 目标：`-s2 top50` ~751µs 尽量下降（终极 400µs）。本任务聚焦「压缩掩码浪费 + 每窗固定
> 开销」。改动仅 `src/compute_big.cuh`（17..32 档档位 1 行；`r19_big_task` L1 段）。
> 口径：`test/mybench -R 32`；s1 = `-s1 tprs 32,128,256 -nodes 3 -replays 2`，
> s2 = `-s2 top50 -replays 3`，us_mean（跨 rank）；各 2~3 次均值。
> 说明：早期在共享主工作副本测得的「PF1 4× 加速」经隔离 worktree 复核为**并发构建
> 污染伪值**（另一 Agent 同时重建 `test/mybench`），已丢弃。

### 1. 诊断（隔离 worktree，R=32）

- `-s2 top50` 为**单波**：每 rank 仅约 7 个窗口（专家平均 count≈18~48），`out_processed`
  最大 ≈2 任务/块；内核墙钟 ≈ 最忙 rank 的单任务耗时。
- 相位（`R24_DIAG_PHASE=1`）：L1 占关键路径最大头（trL1max 典型 36~45 万周期），
  L2 次之；`CM_SKIP_GEMM` 地板 ≈175µs（见方向3 节），故剩余杠杆在 GEMM 本体。
- 掩码：窗口 M 对齐到 16 是 16×16 MMA 的固有粒度；count∈17..31 走 MT2 计 32 行，
  多算的 0~15 行**不增加 B（权重）流量**（B 按 expert 全量、与 count 无关），
  故“行功 6049 vs 4096”主要是 MMA 空转而非带宽项。**方向2 的可行子项收敛为
  「每窗 B 装载延迟隐藏（预取深度）」**，而非削减对齐行。

### 2. 改动

`src/compute_big.cuh:1826-1837`，17..32 档（`count<=32 && g_l1%4==0`）：

```
- r19_l1_big_iso<2, 1, 4>(   // MT2 / B 预取环 1（无重叠）
+ r19_l1_big_iso<2, 2, 4>(   // MT2 / B 预取环 2（一步预取，隐藏装载延迟）
```

A 档（MT2）、N 批（B4）、fragment 路径均不变；仅 B 寄存器预取环深 1→2。
（docs/15/16 当年选 PF1 是为省寄存器；`R19_L1BIG_ISO=__noinline__` 隔离落地后
额外寄存器预算可承受。）**掩码行数、窗口数、win 分布均不变。**

### 3. A/B

| 场景 | 基线 | 本改（PF2） | 变化 |
|---|---:|---:|---:|
| **`-s2 top50` mean** | **750.4**（751.0 / 749.7） | **734.4**（734.8 / 733.9） | **−2.1%** |
| s1 TPR=32 | 489.8（483.7 / 503.2 / 482.5） | 470.1（467.9 / 469.4 / 473.0） | **−4.0%** |
| s1 TPR=128 | 956.2 | 947.0 | −1.0% |
| s1 TPR=256 | 1683.1 | 1681.8 | −0.1% |

s1 三档均不劣（TPR=32 即本档；128/256 走 async 路径，差异在噪声内）。

### 4. 门禁

- `check --smoke`：**PASS 8/8**，6.9s（<10s）。
- `check --all`：**PASS 762/762**，76.3s（`timeout 400` 内）。

### 5. 负结果（同会话，均已否决）

| 试验 | s2 top50 | s1 TPR32/128/256 | 结论 |
|---|---:|---|---|
| 17..32 档 PF3 | 1074.3 | — | 非法（`sp=kk&(kPF-1)` 要求 2 的幂） |
| 17..32 档 PF4 | 739.2 | — | 劣于 PF2 |
| 17..32 档改 r24 A-smem | 737.7 | — | 略劣于 PF2 |
| 小档（≤16）PF4 | 752.7 | — | 中性 |
| 全局 `R19_L1_MT=4` | 753.4 | 496/1146/2050 | TPR128/256 崩塌 |
| 全局 `R24_BPF=4` | 748.4 | 488/1042/1792 | TPR128/256 回退 |
| 全局 `CM_L2_PF=4` | 912.1 | — | 大幅回退（L2 代码膨胀） |
| `R24_KS=16` | 构建失败 | — | A 双缓冲 smem 131328B > 128KB 上限 |
| `KRING_OVERRIDE=12/16` | 746.0 / 744.4 | ~486/942~970/1675~1682 | 仅 −0.5~0.8%，TPR128 微回退，收益不足 |

### 6. 结论

17..32 档 B 预取环 1→2 是**安全且一致的正收益**：`-s2 top50 −2.1%`（750.4→734.4µs）、
均衡 TPR=32 `−4.0%`，其余档不劣；`check --all` 762/762、smoke <10s。
**窗口数/掩码行功均无变化**——本档收益来自每窗 B 装载延迟隐藏。距 400µs 仍远：剩余
主体是 GEMM 本体（≈576µs）与跨 block 分片任务量化（每窗 NS=4 片鸽笼上界），
非「掩码/每窗固定开销」可解。

## [261012_12_43] win=512 + 扁平池 NS4×NK8 L1 split-K（默认关）— 未接近 400µs，仍否决

> 目标：`-s2 top50` ~734µs → 逼近 400µs。假设：win 固定 512（每专家 1 窗）消除按窗
> 重复读权重，再用扁平池 NS=4×NK=8=32 个等代价子任务恢复负载均衡（docs/20）。
> 改动仅 `src/compute_big.cuh`（默认关，关时逐字等价）：给 `r24_l1_big` 加 K 范围 +
> FP32 部分和 epilogue（`:786-793/:944-978`），新增 `r24_l1_ksplit_iso`（`:1042-1090`），
> 并在 `r19_big_task` 的 ksplit 分支 count>64 时改走 R24 快速路径（`:1881-1894`）；
> 新诊断宏 `R19_KSPLIT_NOREDUCE`。开关：`-DR19_BLOCK_KSPLIT=1
> -DR19_BLOCK_KSPLIT_WINMAX=512`（`MT_FORCE_WIN=512` 侧强制 win）。默认关：**零行为变更**。
> 口径：`test/mybench -R 32`；s2 = `-s2 top50 -replays 3`；s1 = `-s1 tprs 32,128,256`。

### 1. 关键发现：瓶颈是 win=512 本身，不是归约

| 口径（s2 top50 mean, µs） | 默认 win=64 | win=512 无 split | win=512+旧K切(r19) | win=512+新K切(r24) |
|---|---:|---:|---:|---:|
| 完整 | **736.8** | ~2700 | 2238.9 | **1991.0** |
| 仅 L1（`CM_SKIP_L2=1`） | 559.4 | 1395.1 | 1968.5 | 1681.5 |
| 仅 L1、去归约（`R19_KSPLIT_NOREDUCE`） | — | — | — | 1454.0 |

- **归约开销 = 1681.5 − 1454.0 = 227.5µs（占仅 L1 的 13.5%）**——与 docs/19 microbench
  一致。用户「归约不该很耗时」的判断 **正确**：归约不是瓶颈。剩余 86.5% 在 K 段的
  GEMM/流水/协议开销。
- **win=512 无 split 的仅 L1（1395µs）已是 win=64（559µs）的 2.5×**：win=512 每专家仅
  4 个 N 片任务，M（行）维不再被窗口切开，热专家（~505 行）的 4 个片任务独占 4 个
  block → 关键路径 ∝ 505 行。**win=64 的 8 个窗天然把 M 切成 8×4=32 个任务**，正是
  均衡所需；win=512 需要 K 切来补回这部分并行。
- 但 K 切补回的并行被**每任务流水开销**吃掉：块级 L1 绝对周期
  （rank17：K 切 1.22M~1.79M vs 默认 0.39M~0.74M，约 2.4×），虽然 **max/min 由
  1.89× 收敛到 1.46×**（均衡确实改善），但每块绝对耗时 ×2.4 → 净回退。

### 2. 减少 NK / 归约优化均无效

| 试验 | s2 top50 mean |
|---|---:|
| win=512 + K 切 NK=8（r24） | 1991.0 |
| win=512 + K 切 NK=4 | 2063.4 |
| win=512 + K 切 NK=8，去归约 | 1763.5（推断 1991−227） |

NK=4（chunk 1536）更差：段变大→并行度下降，且 `r24_issue_a` 的 stage
`__syncthreads` 次数随段数线性放大。用户建议的「非重叠 per-c 部分和」需 ×NK 缓冲
（NK=8 → 4GB/rank）且末段要读 NK× 部分和（≈366µs@1.4TB/s），**比现 reduce 的 227µs 更贵**，
故未采用。

### 3. 门禁 & 结论

- 默认关：`check --smoke` **PASS 8/8**（6.9s）；`check --all` **PASS 762/762**（75.5s）。
- 开（win=512）：`check --smoke`（`MT_FORCE_WIN=512`）**PASS 8/8**；数值正确（FP32 K 累加
  与单块 K 循环顺序不同，非 bit-exact，smoke 容差内）。**未接近 400µs**；两开关保持默认关。
- 结论：win=512 + L1 split-K 设计**方向错误**——它用 K 并行替换了 win=64 已有的 M 并行，
  而每任务流水（smem stage 同步 + 协议 + reduce）随段数 ×8，抵消了均衡收益。真正的杠杆
  是降低 GEMM 本体（≈576µs）与跨 block 分片的任务量化，而非换 win。
- 新增资产（默认关，可复用）：`r24_l1_big` 的 K 范围 + 部分和 epilogue、`r24_l1_ksplit_iso`
  （比原 `r19_l1_ksplit_iso` 在 win=512 快 10%，仅 L1 快 15%）。

## [261012_13_55] 复核：`R19_BLOCK_KSPLIT` 确为「跨 block 扁平池」，回退源是 win=512 本身

> 上一轮怀疑「热专家独占 4 块 → 实为块内 K 切」。本轮用 `R24_DIAG_PHASE` 逐块核对，
> 结论相反：**现有 `R19_BLOCK_KSPLIT` 已经是块级派发**，慢的根因是 win=512。
> 口径同 [261012_12_43]（`-s2 top50 -replays 3`，本机默认 733.9µs）。

| 口径（s2 top50 mean, µs） | 默认(自适应win) | win=512 无切 | win=512 + NS4×NK8 |
|---|---:|---:|---:|
| 完整 | **733.9** | 1992.1 | **2381.7** |
| 纯通信+协议（`CM_SKIP_GEMM=1`） | 171.2 | 486.9 | — |

1. **块级任务分布（rank17, case 806200, `MYBENCH_PHASE_RANK`）**：
   - win=512 无切：仅 **12/32** block 有 L1，`max/min=7.66`；热专家（505 行）的 4 个
     N 片任务**独占 b=24..27**（L1=2.33M cyc）——这才是「独占 4 块」现象。
   - win=512 + NS4×NK8：**32/32** block 有 L1，每块 5~8 个任务，`max/min=1.46`。
     `kseg` 解码（`src/compute.cuh:830-839`）+ 扁平池 FCFS（`:703-706`、`:806-826`）
     确为块级领取，热专家 32 个子任务散到 32 个 block。**均衡要求达成**。
   - 但每块 L1 均值 **1.33M** vs 默认 win=64 的 ~0.4M → 绝对耗时 ×3.3，净回退。
2. **每任务固定开销**是主因：K 段把 K 从 6144 缩到 768（`kL1KChunk`），A-smem 双缓冲
   只剩 `nstage=chunk/kKS=3` 段（`r24_issue_a`/`R24_WAIT_BSM`，`src/compute_big.cuh:898-940`），
   流水未热身即结束；B 直装（`kABSharedBytes=64KB`）与协议开销不随 K 段缩小。故 256 个
   子任务 × ~75µs ≈ 每块 1.3M cyc。win=64 用 **M（窗）并行**得到 24 段长流水，权重
   重复读的代价 < 失去并行的代价（本机权重非瓶颈：`CM_SKIP_GEMM` 仅 171µs）。
3. **归约/atomicAdd 非瓶颈**：win512 base（无 atomicAdd）1992 ≈ ksplit（有 atomicAdd）
   2382 的差距来自 K 切本身；`r19_l1_reduce`（`:445-508`）只占 ~227µs（仅 L1 13.5%）。
   故**未采用**用户建议的「非重叠 `l1_partial[e][s][c]`」：它需 ×NK 缓冲（4.3GB/rank）
   且末段 reduce 读量 ×NK，比现「atomicAdd 累加到单缓冲」更慢。
4. **门禁**：默认关 `check --all` **762/762**（75.4s）、`--smoke` 8/8；开
   `-DR19_BLOCK_KSPLIT=1 -DR19_BLOCK_KSPLIT_WINMAX=512` `check --all` **762/762**（134.8s）、
   `--smoke` 8/8（FP32 累加顺序不同，容差内非 bit-exact）。**未接近 400µs**；保持默认关。
5. **遗留**：win≥256 时 K 段的 `slot_win=1` 令 `pool_slot=e*ring`，需确认 ring 复用语义；
   若要继续此方向，唯一可能有效的是「仅对超阈值 count 的热专家做 K 切（其余保持 NK=1）」，
   以控制任务数（≈win=64 的 56 个）而非 ×4.5。

## [261012_13_46] r19balance：先定位 L1/L2/COMBINE 占比，首个优化=满窗档 N 组批宽 2→4（`-s2 top50` 733.7→714.7µs，−2.6%）

> 口径：`test/mybench -R 32`；s2 = `-s2 top50 -nodes 20 -replays 3`（3 次均值），
> s1 = `-s1 tprs 32,128,256 -nodes 3 -replays 2`。

### 0. 定位（wall 分阶段消融 + 逐 block 相位）

- wall 消融（s2 top50 mean，µs）：完整 **733.7**；`CM_SKIP_L1=1` **391**（L1≈345，**47%**）；
  `CM_SKIP_L2=1` **557**（L2≈179，24%）；`CM_SKIP_GEMM=1` **175**（通信+协议+COMBINE+REDUCE 地板）；
  `MX_SKIP_REDUCE=1` **722**（REDUCE≈14）；`R30_DIAG_NOPUSH=1` **798**（反变慢，不作口径）。
- 逐 block（`-DR24_DIAG_PHASE=1 MYBENCH_PHASE_RANK=17`，case 806200）：L1 累计最高 **730k cyc**
  （b6）、L2 最高 **497k**；最忙 block 的 L1/L2 = 730k/497k，`max/avg(L1)≈1.53×`。
  block0 时间线 `L2tail=1.01M cyc` **并非纯 L2**——它含第 2 任务的 dispatch+L1+L2+COMBINE+等待。
- **结论：占比最大是 L1 GEMM（47%），L2（24%）与 COMBINE+协议+REDUCE（24%）次之。**
  「热 rank L2tail≈1.02M」是相位包含关系导致的误读，COMBINE 排空不是最大项。

### 1. 首个优化：满窗（49..64 行）档 N 组批宽 2→4

- 机制：49..64 档 `kMT=4`、`g_l1=(2I/block_n)/nslice=4`；原 `R19_L1_BATCH=2` 使 g0 循环
  分两次，A fragment **被重复 stage 一遍**。本档改走新宏 `R19_L1_BATCH64=4`（默认 4），
  一次覆盖 4 组 N → A 只装一遍。`g_l1` 非其倍数（NS=8→`g_l1=2`）时自动回落原 batch。
- 改动：`src/compute_big.cuh`（新增 `R19_L1_BATCH64`；49..64 档两处 r24 调用按 `g_l1` 选批宽）。

| 口径（mean, µs） | 基线 | BATCH64=4 | 变化 |
|---|---:|---:|---:|
| s2 top50 | 733.7 | **714.7** | **−2.6%** |
| s1 TPR32 | 472 | 466 | ~0 |
| s1 TPR128 | 948 | 953 | +0.5% |
| s1 TPR256 | 1678 | 1681 | +0.2% |

### 2. 门禁

- `check --smoke` **8/8**（6.4s，<10s）；`check --all` **762/762**（76s）。
- s1 三档均不回退 >5%；s2 稳定 −2.6%（两次 713.9/715.5）。

### 3. 负结果（已回退，未提交）：L1 M 切

- 按「L1 大 → L1 行切（M 切，无归约）」实现跨 block M 切（`R19_BLOCK_MSPLIT`，默认关；
  16 行对齐、owner 共享 pull、末段发 `l1_prog`）。`check --smoke` 数值正确，但
  **s2 733.7→1057.6（+44%）、s1 TPR32 +60%**。根因：每 32 行段重读**整片权重（B）**，
  与 win=32 / NS=8 同源——**细化 M 反而放大权重流量**，确认 L1 为权重带宽主导。
  故 L1 的可行方向是**提高每遍权重的 M 复用**（更大有效 M），而非更细切分。

## [261012_14_35] r19balance：`dispatch_pull_body` 去函数级循环不变量局部缓存 — local 2584→2488B、`-s2 top50` 715.2→709.7µs（−0.76%）

> 目标：`-s2 top50` 714.7µs → 500µs。本轮聚焦降低 **calling function（`mega_kernel`）**
> 跨两个 `__noinline__` 任务体（`r19_small_task`/`r19_big_task`）的寄存器保活/溢出，
> 给任务体/流水腾寄存器。
> 改动仅 `src/compute.cuh`（`dispatch_pull_body`）：删除 **18 个函数级循环不变量局部缓存**
> （`self`/`R`/`win_cap`/`pool_ptr`/`bar_flags`/`task_counter_idx`/`l2_done_idx`/
> `task_ready_idx`/`out_processed`/`dbg`/`max_spins`/`l1_chk`/`l2_chk`/`rep`/`n_reps`/
> `epoch`/`prefix_ptr`/`pt`/`frag_a`/`frag2`），在每个**使用点**直接读 `a->...`（`s_a`, smem）
> 或 `a->rwk->lf....`；`ring`→`kRing`；`rep=0`/`n_reps=1`→`constexpr`。行为逐字不变
> （同值、同顺序；两处对数组的 `!= nullptr` 守卫恒真，去除后消除告警、语义等价）。

### 1. 内核资源（`MT_DUMP_RES=1`，`-s1 -tprs 16 -nodes 1 -replays 1`）

| 版本 | regs | local |
|---|---:|---:|
| 基线（S-SHRINK，HEAD `0329d40`） | 256 | **2584 B** |
| 本改（去缓存） | 256 | **2488 B（−96 B，−3.7%）** |

`regs` 仍 256（顶格）；这 21 个不变量（多为 8B 指针/int64，本就存于 smem `s_a`）不再作为
独立局部跨 noinline 调用保活，栈帧/local 下降。

### 2. A/B（隔离 worktree；base/new `.so` 逐轮交替，us_mean 跨 rank）

| 场景 | 基线 | 本改 | 变化 |
|---|---:|---:|---:|
| **`-s2 top50` mean（3 轮交替）** | 715.21（714.96 / 716.45 / 714.23） | **709.74（709.66 / 711.15 / 708.40）** | **−0.76%** |
| s1 TPR=32（6 次合并） | 472.4 | 475.4 | +0.6%（噪声内） |
| s1 TPR=128 | 941.6 | 939.8 | −0.2% |
| s1 TPR=256 | 1678.4 | 1677.8 | ~0% |

s2 稳定 −0.76%（3/3 轮均改善）；s1 三档在噪声内（TPR=32 单次波动 ±13µs，大于差异）。
基线 local=2584 与 HEAD 复现一致。

### 3. 门禁

- `check --smoke` **PASS 8/8**，6.7s（<10s）。
- `check --all` **PASS 762/762**，75.7s（`timeout 400` 内）。

### 4. 结论与遗留

去缓存后 `local 2584→2488B`、`-s2 top50` 稳定 −0.76%、s1 不回退。收益有限（被去缓存的量
本就在 smem，非溢出主因），但为后续「给任务体/流水腾寄存器」留出空间。距 500µs 仍远；
剩余主体是 L1 GEMM（≈47%）与跨 block 分片任务量化。

## [261012_15_39] r19balance：精简 `MegaArgs` 成员（去 `l1_k_out`/`l2_k_out`/`y`/计数）— local 2488→2288B、`-s2 top50` 709.9→704.6µs（−0.75%）

> 目标：`-s2 top50` 向 500µs 推进，同时精简 `src/compute.cuh::MegaArgs`（减 `s_a` smem
> 与使用点寄存器/指令）。逐字段独立 gated，删后跑门禁与 A/B；会破坏「单 .so 双路径」
> 或 torch 语义的字段保留。

### 1. 逐字段处理结果

| 字段 | 处理 | 依据 |
|---|---|---|
| `l1_k_out`(=I=2048) | **删**，全部使用点改常量 `kIntermediate` | test 传 `kIntermediate`；torch 冻结 `FROZEN_INTERMEDIATE_HIDDEN=2048` |
| `l2_k_out`(=H=6144) | **删**，改常量 `kHidden` | test 传 `kHidden`；torch 冻结 `FROZEN_HIDDEN=6144` |
| `y` | **删**；移除 `MXK_FUSE_REDUCE` 融合分支（默认关）及 `to_mega_args` 拷贝 | 设备侧仅 fuse 分支读；独立 REDUCE 核用 `MegaLaunchArgs::y`，保留 |
| `exec_ctr`/`dbg_count` | **删**；`mega_kernel` 入口恒 `ctr=nullptr` | `#if MX_DBG_COUNT` 默认 0，无功能影响 |
| `pool_ptr` | **留** | torch 传 `nullptr` → 用 `sym+pool_off`；test 传 `rwk->ld.expert_tokens`。二者不等，派生会改 torch 行为 |
| `prefix_ptr` | **留** | 同上：torch 用 `sym+prefix_off`，test 用 `rwk->ld.prefix_tbl` |
| `in_bf16` | **留** | torch 运行时按 `SymmBuffer::L.in_bf16` 区分 bf16/fp8（单 .so 双路径），编译期化会破坏 |
| `slot_capacity` | **留** | test `--max-tokens` 可变、torch `cap` 可变，非固定 |

`l1_k_out`/`l2_k_out` 共 ~175 处使用点（`a->`/`S.a->`）逐点替换为常量；字段声明、
`MegaLaunchArgs`/`to_mega_args`/`make_args`/torch `make_mega_args` 同步删除。行为逐字不变。

### 2. 内核资源（`MT_DUMP_RES=1`，`-s1 -tprs 16 -nodes 1 -replays 1`）

| 版本 | regs | local | smem(full) | smem(empty) |
|---|---:|---:|---:|---:|
| 基线（HEAD `06bb622`） | 256 | **2488 B** | 66160 | 592 |
| 本改 | 256 | **2288 B（−200 B，−8.0%）** | 66144 | 576 |

### 3. A/B

| 场景 | 基线 | 本改 | 变化 |
|---|---:|---:|---:|
| **`-s2 top50` mean（3 轮）** | 709.87 | **704.57（704.68 / 704.56 / 704.57）** | **−0.75%** |
| s1 TPR=32 | 457.16 | 461.87 | +1.0%（噪声） |
| s1 TPR=128 | 940.07 | 942.93 | +0.3%（噪声） |
| s1 TPR=256 | 1678.47 | 1626.16 | **−3.1%** |

### 4. 门禁

- `check --smoke` **PASS 8/8**，6.5s（<10s）。
- `check --all` **PASS 762/762**，72.7s（`timeout 400` 内）。

### 5. 结论与遗留

删 4 个设备字段使 `local 2488→2288B`、`s_a` smem −16B，`-s2 top50` 稳定 −0.75%、
s1 TPR256 −3.1% 且其余不回退。`pool_ptr`/`prefix_ptr`/`in_bf16`/`slot_capacity` 因
torch 语义或运行期灵活性保留。距 500µs 仍远，主体仍为 L1 GEMM 与跨 block 分片。

## [261012_15_53] r19balance：移除 `MegaArgs` 6 个 `*_idx` 字段（改编译期常量）— `s_a` smem −48B、`-s2 top50` 704.6→703.7µs（中性）

> 目标：继续精简 `MegaArgs`（上一条 [261012_15_39] 已去 `l1_k_out`/`l2_k_out`/`y`/计数）。
> 本轮把 6 个同步旗索引改为 kernel 侧编译期常量（host/torch 不再传），行为逐字不变。

### 1. 改动

- `src/common.cuh`：新增 `constexpr int64_t kBarXreleaseIdx=65`/`kBarSliceFlagsIdx=67`/
  `kBarL1ProgressIdx=1091`/`kBarL2DoneIdx=2115`/`kBarTaskReadyIdx=2499`/
  `kBarTaskCounterIdx=2501`（镜像 `include/deep_gemm/layout/mega_moe.cuh` 的 `kBar*`；
  kernel 不 include 该头，避免引入 cute 依赖）。
- `src/compute.cuh`：删 `MegaArgs` 的 6 个 `*_idx` 字段；`dispatch_pull_body` 使用点改常量。
- `src/compute_big.cuh`（7 处 `S.a->`）、`src/kernel.cuh`（`a.*` → 常量）。
- `src/megakernel.h`/`src/megakernel.cu`/`test/mega_common.cuh`/`torch/csrc/apis/mega.hpp`：
  同步删字段与赋值；`to_mega_args`/`make_args`/`make_mega_args` 不再拷贝；
  `mega_reduce_kernel` 的 `task_ready_idx` 直接用常量。
- `test/mega_common.cuh`：加 6 条 `static_assert(kBar*Idx == deep_gemm::layout::kBar*)`
  保证与布局头同源（torch 侧不 include 设备头，故不设断言，仅在常量处注明来源）。

注：任务书给出的 `kBarTaskReadyIdx=1018`/`kBarTaskCounterIdx=1020` 为**旧硬编码值**；
当前布局头（`kMegaMoeMaxGridBlocks=64`，slice/l1 区各 1024、l2done 256）实为
`2499`/`2501`，否则 `static_assert` 失败且旗槽错位。本轮按布局头实际值冻结。

### 2. 内核资源（`MT_DUMP_RES=1`，`-s1 -tprs 16 -nodes 1 -replays 1`）

| 版本 | regs | local | smem(full) | smem(empty) |
|---|---:|---:|---:|---:|
| 基线（HEAD `a46c998`） | 256 | 2288 B | 66144 | 576 |
| 本改 | 256 | **2288 B** | **66096（−48B）** | **528（−48B）** |

`local` 不变（2288B）；`s_a` smem 恰减去 6×int64=48B。

### 3. A/B（us_mean 跨 rank）

| 场景 | 基线 | 本改 | 变化 |
|---|---:|---:|---:|
| **`-s2 top50` mean** | 704.57 | **703.67** | **−0.13%（噪声内）** |
| s1 TPR=32 | 462 | 460.28 | ~0%（噪声） |
| s1 TPR=128 | 943 | 912.02 | −3.3%（噪声方差大） |
| s1 TPR=256 | 1626 | 1619.33 | −0.4%（噪声内） |

### 4. 门禁

- `check --smoke` **PASS 8/8**，6.77s（`timeout 10` 内）。
- `check --all` **PASS 762/762**，73.4s（`timeout 400` 内）。

### 5. 结论与遗留

纯结构精简：`s_a` smem −48B、使用点少 6 次跨 noinline 结构体读，`local`/`regs` 与
`-s2 top50` 均在噪声内。收益有限（索引本就在 smem），但为后续继续瘦身 `MegaArgs`
铺路。距 500µs 仍远，主体仍为 L1 GEMM 与跨 block 分片任务量化。

## [261012_16_05] r19balance：`pool`/`prefix` 移入 `rwk->ld`（rank 私有）— `MegaArgs` −4 字段、`s_a` smem −32B、`-s2 top50` ≈中性

> 目标：继续精简 `MegaArgs`（上一条 [261012_15_53] 已把 6 个旗索引改编译期常量）。
> 本轮把 DISPATCH pool（expert_tokens）与前缀表（prefix_tbl）从「sym scratch 偏移 /
> 可选外部指针覆盖」彻底改为 **本 rank 私有 `rwk->ld`**：kernel 恒取
> `&a->rwk->ld.expert_tokens[0][0][0]` / `&a->rwk->ld.prefix_tbl[0][0]`，
> task_setup 也恒写 `rwk->ld.prefix_tbl`。行为逐字不变（pool/prefix 本就是 rank 内读写）。

### 1. 改动

- `src/compute.cuh`：删 `MegaArgs` 的 `pool_ptr`/`prefix_ptr`/`prefix_off`/`pool_off`
  四字段；`dispatch_pull_body` 的 prefix/pool 基址改用 `rwk->ld`。
- `src/megakernel.h`/`src/megakernel.cu`：删 4 字段与 `to_mega_args` 拷贝。
- `src/kernel.cuh`：`task_setup_body` 的 `prefix_ptr` 实参改 `&a.rwk->ld.prefix_tbl[0][0]`，
  `prefix_off` 传 0（不再用 sym 回退）。
- `src/compute_big.cuh`：`R19Sched::pool_base` 注释同步。
- `test/mega_common.cuh`/`torch/csrc/apis/mega.hpp`：删对应赋值（torch 不再依赖 sym 的
  expert_tokens/prefix）。sym 布局（`mega_moe.cuh`/`host_layout.cuh`/`layout.hpp`）**保留**
  expert_tokens/prefix_tbl 槽（仅不再被 kernel 使用；见遗留）。

### 2. 内核资源（`MT_DUMP_RES=1`，`-s1 -tprs 16 -nodes 1 -replays 1`）

| 版本 | regs | local | smem(full) | smem(empty) |
|---|---:|---:|---:|---:|
| 基线（HEAD `097b25b`） | 256 | 2288 B | 66096 | 528 |
| 本改 | 256 | 2288 B | **66064（−32B）** | **496（−32B）** |

`local`/`regs` 不变；4×8B 字段从 smem `s_a` 移除，恰 −32B。

### 3. A/B（us_mean 跨 rank）

| 场景 | 基线 | 本改 | 变化 |
|---|---:|---:|---:|
| **`-s2 top50` mean（3 轮）** | 703.67 | **701.82（700.61 / 701.49 / 703.36）** | ~0%（噪声内，略优） |
| s1 TPR=32 | 460 | 464.13 | +0.9%（噪声） |
| s1 TPR=128 | 912 | 911.57 | ~0% |
| s1 TPR=256 | 1619 | 1614.93 | −0.3%（噪声） |

### 4. 门禁

- `check --smoke` **PASS 8/8**，6.59s（`timeout 10` 内）。
- `check --all` **PASS 762/762**，72.9s（`timeout 400` 内）。

### 5. 结论与遗留

结构精简：`s_a` smem −32B、`MegaArgs` −4 字段；性能中性（`-s2 top50` 略优 0.26%）。
sym scratch 中的 `expert_tokens`/`prefix_tbl` 槽**保留未删**（`L.expert_tokens_off`/
`L.prefix_tbl_off` 及 `host_layout`/`layout.hpp` 仍在），仅 kernel 不再读写，属可省内存的
后续项；torch 公有视图 `mega.l1_acts`（原指向 sym expert_tokens）随之不再反映 kernel pool
（仓内无消费者）。距 500µs 仍远，主体仍为 L1 GEMM 与跨 block 分片任务量化。

## [261012_16_36] r19balance：内部 sym scratch 并入 `unk_data`（`rd`）— 删 4 个 `*_off`、`local` −32B、`-s2 top50` ≈中性

> 目标：把**非对外**的 sym scratch 字段并入 `unk_data`（`src/flags_defs.cuh` 的
> `ranks_data`），仿 `ranks_data` 按最大使用量声明数组；kernel 改走
> `a->unk->rd` / `a->unk_peers`。**对外输入视图 `x`/`topk_idx`/`topk_weights`
> 仍留在 sym（不变）**。`src_index_table`/`expert_recv_counts`/`combine_rows`
> 本就在 `rd`，本轮去重后 kernel 恒用 `rd`。

### 1. 改动

- `src/flags_defs.cuh`：`ranks_data` 追加 `token_route_weights[kGrid*kMaxTPR]`、
  `token_src_meta[kGrid*kMaxTPR*3]`、`prefold[kMaxTPR][kHidden]`（`alignas(256)`）。
- `src/compute.cuh`/`src/megakernel.{h,cu}`：删 `MegaArgs`/`MegaLaunchArgs` 的
  `w_off`/`meta_off`/`prefold_off`/`combine_rows_off`（及 `to_mega_args` 拷贝）；
  新增诊断位 `skip_combine`（取代原 `combine_rows_off<0` 语义）。
- `src/compute_big.cuh`：staging `wt/meta` 改 `a->unk->rd.token_route_weights/
  token_src_meta`；pull 源 `prefold` 改 `unk_peers[src]->rd.prefold`（fp8 `x` 仍走
  `peer_bases`+`x_off`）；`cm_build_ctx`/`r19_tail_common` 的 COMBINE 开关改 `combine_on`。
- `src/kernel.cuh`：`x_fold_body` 写 `a.unk->rd.prefold`；`mega_reduce_kernel` 删 `rows_off`。
- `src/combine.cuh`：`CombineCtx.rows_off`→`combine_on`；`cmb_rows_compat` 恒取
  `unk_peers[src]->rd`；`reduce_tile_body`/`reduce_body` 删 `combine_rows_off`
  （`combine_rows` 恒从 `remote=rd` 读）。另含工作区既有的 `MXK_FUSE_REDUCE`（默认 0）
  脚手架。
- `src/multi_compile/kernel_reduce.cu`、`test/mega_common.cuh`、`torch/csrc/apis/mega.hpp`：
  同步删对应形参/赋值。
- sym 布局（`mega_moe.cuh`/`host_layout.cuh`/`layout.hpp`）**保留**该部分 scratch 槽
  （仅不再被 kernel 使用，降低风险；见遗留）。

### 2. 内核资源（`MT_DUMP_RES=1`，`-s1 -tprs 16 -nodes 1 -replays 1`）

| 版本 | regs | local | smem(full) | smem(empty) |
|---|---:|---:|---:|---:|
| 基线（HEAD `5362f49`） | 256 | 2288 B | 66064 | 496 |
| 本改 | 256 | **2256（−32B）** | **66032（−32B）** | **464（−32B）** |

### 3. A/B（us_mean 跨 rank，同会话）

| 场景 | 基线 | 本改 | 变化 |
|---|---:|---:|---:|
| **`-s2 top50` mean（3 轮）** | 699.52 | **701.50** | +0.28%（噪声内） |
| s1 TPR=32 | 451.84 | 449.22 | −0.6%（噪声） |
| s1 TPR=128 | 932.98 | 900.97 | −3.4%（噪声/利好） |
| s1 TPR=256 | 1623.17 | 1620.23 | −0.2%（噪声） |

### 4. 门禁

- `check --smoke` **PASS 8/8**，5.99s（`timeout 10` 内）。
- `check --all` **PASS 762/762**，73.72s（`timeout 400` 内）。
- `MACA_PATH=/opt/maca scripts/build.sh so check mybench` 通过。

### 5. 结论与遗留

行为逐字不变的结构精简：`MegaArgs` −4 个 `*_off` 字段、`s_a` smem −32B、`local` −32B；
性能中性（`-s2 top50` +0.28% 在噪声内，s1 TPR=128 略优）。sym 中该部分 scratch 槽**保留未删**
（`L.token_route_weights_off`/`token_src_meta_off`/`prefold_off`/`combine_rows_off` 仍计算），
仅 kernel 不再读写，属可省内存/可继续收紧的后续项。距 500µs 仍远，主体仍为 L1 GEMM 与
跨 block 分片任务量化。

## [261012_16_47] r19balance：删 `MegaArgs` 的 `w1/w2/ffn_hidden/l2_out`（kernel 从 `rwk->ld` 派生）— `local` −16B、`s_a` smem −32B、`-s2 top50` ≈中性

> 目标：把最后 4 个 `MegaArgs` 显存指针字段删除。它们都是 `local_data`（本 rank
> 私有 `rwk->ld`）成员：`w1[kEPR][2I][H]`、`w2[kEPR][H][I]`、
> `ffn_hidden[kEPR*kRing][kMaxTPR][I]`、`l2_out[kGrid][kMaxTPR][H]`。kernel 一律
> 从 `a->rwk->ld` 取基址，host/torch 不再传入。对外 API（权重入参、`x`/`topk`
> 视图）名称/语义不变。

### 1. 改动

- `src/compute.cuh`：删 `MegaArgs::w1/w2/ffn_hidden/l2_out`（4 个指针）。
- `src/compute_big.cuh`：所有 `S.a->w1/w2/ffn_hidden/l2_out` 使用点（共 46 处）
  改 `&S.a->rwk->ld.{w1,w2,ffn_hidden,l2_out}[0][0][0]`（类型对：w1/w2 `const uint8_t*`、
  ffn_hidden `uint8_t*`、l2_out `uint16_t*`）。
- `src/megakernel.{h,cu}`：删 `MegaLaunchArgs` 4 字段与 `to_mega_args` 拷贝。
- `test/mega_common.cuh`：删 `a.w1/w2/ffn_hidden/l2_out` 赋值（`d.w1/d.w2/d.ffn/d.l2`
  仍指向同一 `ld`，权重填充/清零/校验不变）。
- `torch/csrc/apis/mega.hpp`：删 `a.*` 赋值；停独立分配 `ffn_hidden`/`l2_out`
  （改用 `rwk->ld`）；`w1/w2` 改为首次 launch 前一次性 D2D 拷进 `rwk->ld.w1/w2`
  （按来源指针缓存，静态一次成本）。
- `torch/csrc/flags_bridge.cu`：新增 `mxk_rwk_w1_off()/mxk_rwk_w2_off()`（`rwk` 基址
  到 `ld.w1/w2` 的字节偏移）；`torch/csrc/python_api.cpp`：校验权重 numel 与
  `ld.w1/w2` 一致后再拷。

### 2. 内核资源（`MT_DUMP_RES=1`，`-s1 -tprs 16 -nodes 1 -replays 1`）

| 版本 | regs | local | smem(full) | smem(empty) |
|---|---:|---:|---:|---:|
| 基线（HEAD `524b86d`） | 256 | 2256 B | 66032 | 464 |
| 本改 | 256 | **2240（−16B）** | **66000（−32B）** | **432（−32B）** |

### 3. A/B（同会话，先 stash 本方改动重建基线再实测）

| 场景 | 基线 | 本改 | 变化 |
|---|---:|---:|---:|
| **`-s2 top50` min（3 轮）** | 712.68 / 713.10 / 723.37（均 716.4） | **725.06 / 712.24 / 711.07（均 716.1）** | ≈0%（噪声内） |
| s1 TPR=32（2 轮） | 448.94 / 436.14 | 463.49 / 448.09 | +1.4%（噪声） |
| s1 TPR=128（2 轮） | 921.34 / 896.13 | 930.09 / 905.86 | +1.1%（噪声） |
| s1 TPR=256（2 轮） | 1607.55 / 1606.53 | 1607.98 / 1600.17 | ≈0% |

> 注：本会话 `-s2 top50` 整体基线（712–724µs）显著高于 `perf.md` 历史记录
> （699–702µs），系机器负载/会话差异；A/B 以同会话交替口径为准，结论中性。

### 4. 门禁

- `check --smoke` **PASS 8/8**，6.68s（`timeout 10` 内）。
- `check --all` **PASS 762/762**，73.33s（`timeout 400` 内）。
- `MACA_PATH=/opt/maca scripts/build.sh so check mybench` 通过。

### 5. 结论与遗留

行为逐字不变的结构精简：`MegaArgs` 再 −4 个指针字段、`s_a` smem −32B、`local` −16B；
性能中性（同会话 A/B 噪声内）。**torch 侧改动本环境无 torch 无法编译验证**，仅静态审查
（API 名称/语义不变；`mxk_rwk_w1_off/w2_off` 新增偏移导出），待重出 whl 时验证。


## [261012_18_58] r19balance/impl1：fp4(e2m1) 权重 + packed 预取环（`-DMX_FP4_W=1`，默认关）— `-s2 top50` 698→582µs（−16.7%），B 暴露 216→134µs（−38%）

> 目标：W1/W2 以 fp4 打包（super-tile 512B、每 lane 8B），kernel 用
> `__builtin_mxc_cvt_pk8_f4tof8_scale_lo` 解包回 fp8 fragment 复用现有 fp8 MMA
> （指令不变）→ 权重 HBM 字节减半。上版原型「cvt 放进 B 预取环」净回退
> （`-s2 top50` 695→877µs，B 暴露 216→427µs），本版定位并修复。

### 1. 根因定位（同会话微基准 + 消融）

- cvt 吞吐微基准（本机 MetaX C600-UL，32 SM / 1.6GHz，依赖链）：
  128×256（16 warps/SM）362 Gcvt/s；32×256（**= 本 kernel 的 4 warps/SM**）123 Gcvt/s
  → cvt 为**延迟受限**，低占用下吞吐仅 1/3。
- `-DMX_FP4_NOCVT`（保留 8B 装载、只删 cvt）`-s2 top50` = 660µs < 基线 695µs
  → 多出的全部代价是 cvt 的调度暴露，与装载宽度/字节无关（`ldx_b64` vs `ldx_b128` 非主因）。
- `MT_DUMP_RES=1`：base regs=256/local=2240；fp4 regs=256/local=2252 —— 无额外溢出。

### 2. 修法（`MX_FP4_RING`，默认 1；仅 `MX_FP4_W=1` 时生效）

- 预取环由 `R19Frag2 b_pair[k][batch]`（16B/项，cvt 在环内）改为
  `uint64_t b_pk[k][batch]`（8B/项，仅装载）；消费点按 k-step 批量
  `R19Frag2 bexp[kBatch] = r19_expand_pk(b_pk)` 再喂 MMA。
- 应用到 r19_l1_big / r24_l1_big / r19_l2_tile / r24_l2_big 四条路径。
- 新增 `r19_load_b_pk` / `r19_expand_pk`；`R19_LDB_PK` 与 `R24_DIAG_NOB` 兼容。

### 3. A/B（同会话交替，us_mean）

| 场景 | 基线 | fp4+RING | 变化 |
|---|---:|---:|---:|
| `-s2 top50` mean（3 轮） | 698.2 | **581.6** | **−16.7%** |
| `-s2 top50` B 暴露（`R24_DIAG_NOB`） | 216.4 | **134.0** | **−38%** |
| s1 TPR=32（nodes3 replays2） | 446.9 | 396.9 | −11.2% |
| s1 TPR=64 | 620.1 | **494.2** | **−20.3%** |
| s1 TPR=128 | 915.3 | 890.9 | −2.7% |
| s1 TPR=256 | 1597.3 | 1562.5 | −2.2% |

### 4. 门禁

- `MACA_PATH=/opt/maca scripts/build.sh so check mybench` 通过（默认与 fp4 两种）。
- `timeout 10 scripts/run.sh check --smoke` **PASS 8/8**，6.4s。
- 默认 `MX_FP4_W=0`：逐字等价基线（`#if` 全关）。

### 5. 结论与遗留

- fp4 权重经 packed 预取环后真正省带宽：B 暴露 216→134µs 落入目标窗口 110~145µs；
  s2 698→582µs。**距 500µs 仍差 ~82µs**（NOB 地板 450µs 已占主要）。
- 预取深度 A/B：`MX_FP4_W=1 + R24_BPF=4 + R19_L1_PF=8` 可再降至 ~567µs，
  但会同时改变默认 fp8 路径（历史 TPR128/256 回退），需做成 fp4 条件默认，未纳入。
- 权重 fp4 量化/reorg 仅在 bench/check 侧实现；torch 侧未接入（本环境无 torch）。

## [261012_19_24] r19balance/impl1：fp4 条件默认加深 B 预取环（`#if MX_FP4_W`）— `-s2 top50` 583.7→569.9µs（−2.4%），fp8 默认逐字不变

> 目标：把上节遗留的「`MX_FP4_W=1 + R24_BPF=4 + R19_L1_PF=8` ≈567µs」做成 **仅 fp4
> 路径** 生效的条件默认。直接改默认会同时改变 fp8 路径（历史大档回退），本版用
> `#if MX_FP4_W` 覆盖 `R24_BPF`/`R19_L1_PF` 的 `#ifndef` 默认值。

### 1. 改动（`src/compute_big.cuh`）

- `R24_BPF` 默认 `2→4`、`R19_L1_PF` 默认 `2→8`，**仅当 `MX_FP4_W`**；`#else` 分支
  与 `-D` 覆盖语义保持原文逐字不变。`R24_BPF` 经 `kR24BPF` 同时作用于 L1/L2 的
  kAsync（`count>64`）与尾块路径；`R24_DIAG_NOB` 消融确认 fp4 下 B 暴露仍为 ~121µs
  （NOB 地板 449µs），本次收益来自更深的 packed 环覆盖剩余 B 延迟。

### 2. A/B（同会话交替，us_mean；s2=top50 replays5、s1=nodes3 replays3）

| 口径 | fp4 基线 (BPF2/PF2) | fp4 条件默认 (BPF4/PF8) | 变化 |
|---|---:|---:|---:|
| `-s2 top50` mean（3 轮） | 583.7 | **569.9** | **−2.4%** |
| s1 TPR=32 | 392.2 | 389.2 | −0.8% |
| s1 TPR=64 | 505.5 | 494.8 | −2.1% |
| s1 TPR=128 | 895.6 | 894.4 | ≈0 |
| s1 TPR=256 | 1562.2 | 1579.0 | +1.1% |

### 3. 门禁

- `MACA_PATH=/opt/maca scripts/build.sh so check mybench`（`MX_FP4_W=0/1` 两种）通过。
- `timeout 10 scripts/run.sh check --smoke` **PASS 8/8**，6.2s（<10s）。
- `MX_FP4_W=0` 行为逐字不变：`static_assert` 探针确认默认仍 `R24_BPF==2 && R19_L1_PF==2`；
  `-DMX_FP4_W=1 -DR24_BPF=2 -DR19_L1_PF=2` 亦确认 `-D` 覆盖仍生效。

### 4. 结论与遗留

- fp4 下 `-s2` 583.7→569.9µs（−2.4%）；距 500µs 仍差 ~70µs（NOB 地板 449µs 为主体）。
- s1 TPR=256 +1.1%：大窗 kAsync 亦随之用 BPF=4。可选后续把深度改由
  `S.a->win_cap<=64` 运行期门控（大窗保持 2），需复制 kAsync 实例，本轮未纳入。
- torch 侧改动（工作副本内 flags_bridge/init）单列，未随本提交验证。

## [261013_00_02] agent/peerbuf/stage3：组内均衡调度 + 远端 helper 计算（`-DMX_GROUP_SCHED=1`，默认关）— `-s2 top50` 695→571µs（−17.8%）

> 目标：把 `-s 2 -top 50`（TK 实录不均衡）的 fp8 路径 mean 从 ~690µs 压向 500µs。
> 做法：**交织组**（组 g = rank%8，member = rank/8）内，按确定性 LPT 把本组 32 个专家的
> **窗口**重分配给组内 4 个 rank；非 owner 窗口由 helper 远端拉取 A（读 owner 的
> `expert_recv_counts`/`src_index_table`）、用本组权重 `w1g/w2g`（ge 索引）计算，L2 结果
> 经 COMBINE 推回源 rank，`combine_sums` 协议不变。默认 `MX_GROUP_SCHED=0` 逐字节不变。

### 1. 改动

- 新增 `src/group_sched.cuh`：交织组几何 + 组倾斜门控（`max·kGE > 3·Σ`）+ 桶式无 scratch
  确定性 LPT（cost=ceil(rows/16)，cost 降序、(ge,w) 升序；平手优先 owner 再低序号）。
- `src/flags_defs.cuh`：`kGroupRanks` 可用 `-DMX_GROUP_RANKS` 覆盖（默认 4）。
- `src/compute.cuh`（`dispatch_pull_body`）：gs 激活时按本 rank 窗口序号 seq 建任务表
  （`pool_slot=seq%kGSRing`、`ring=kGSRing=64`、`gen=epoch+seq/ring`），任务尾 `l2_done`
  按 seq 判定复用；均衡组（不倾斜）回退逐字基线路径。
- `src/compute_big.cuh`：`r19_pull_flags`/任务体新增 `owner`/`seq`；gs 下 `src_index_table`
  用 owner 的 FG 表且专家维索引折 `le=ge%kEPR`；权重索引用组内专家序 `ge`。
- `test/check.cu`：`check_l1l2` 按同一 gs 门控/LPT 定位 rank0 的正确 pool/frag 槽。
- `src/host_layout.cuh`：`MX_GROUP_SCHED` 下动态 smem=40KB（gs 表；<48KB 默认上限）。

### 2. A/B（同会话，R=32；s2=top50 nodes20 replays2 取 min；s1=nodes20 replays2）

| 口径 | 默认 (MX_GROUP_SCHED=0) | 阶段3 (MX_GROUP_SCHED=1) | 变化 |
|---|---:|---:|---:|
| `-s2 top50` mean | 695.2 | **570.9** | **−17.9%** |
| s1 TPR=16 | 315.1 | 323.7 | +2.7% |
| s1 TPR=32 | 417.1 | 431.4 | +3.4% |
| s1 TPR=256 | 1603.7 | 1611.4 | +0.5% |

### 3. 门禁

- `KERNEL_CFLAGS="-DMX_GROUP_SCHED=1" scripts/build.sh so mybench check` 通过。
- `timeout 10 scripts/run.sh check --smoke` **PASS 8/8**（7.0s，<10s）。
- `timeout 400 scripts/run.sh check --all` **PASS 762/762**（49.7s）——含全部 TK 实录 launch
  （均衡 TPR 全表 + 762 用例），REDUCE/L1L2/graph 全通过。
- 默认 `MX_GROUP_SCHED=0`：`check --smoke` PASS 8/8、`-s2 top50` mean 695.2µs（与基线一致）。

### 4. 结论与遗留

- `-s2 top50` 695→571µs（−17.9%）；距 500µs 仍差 ~70µs。
- 均衡 TPR 路径回退 +0.5~3.4%（组倾斜判定需读 owner FG 计数 + gs 表构建），因 `MX_GROUP_SCHED`
  默认关，生产不受影响；后续可用「是否曾出现倾斜」的跨 launch 记忆来对均衡负载零成本。
- `MX_GROUP_RANKS=8`（组 8）实测更差（598µs）：权重复制 2×（rwk 2.75GB）、远端读更多，
  收益被抵消；默认 4。
- 未达 <500µs：wall 由最忙 block 的**单个大窗任务**决定，rank 级重分配有上限；后续需配合
  更细粒度切片或 K 切（`R19_BLOCK_KSPLIT`）进一步缩短单任务。

## [261013_00_16] agent/peerbuf/stage3b：组内均衡调参扫描 — `-s2 top50` 564.6µs（group-4 最优），550µs 目标需更细粒度单任务

> 目标：在 stage3（571µs）基础上继续压低 `-s2 top50`。本轮做参数扫描 + 廉价优化，
> 默认 `MX_GROUP_SCHED=0` 逐字不变；以下均为 gs 路径（`-DMX_GROUP_SCHED=1`）。

### 1. 改动（默认保持基线）
- 组倾斜门控阈值参数化 `MX_GS_SKEW_NUM`（默认 3）；代价模型 `MX_GS_COST_MODEL`
  （0=ceil(rows/16) 默认；1=rows；2=+4 固定项）；单 rank 窗口表容量 `MX_GS_MEMBER_CAP`。
- 组计数探测的并行读**顺带缓存原值**，前缀和改为 smem 原地转换（省一遍串行远端读）。
- `gs_cap` 放宽到 `nGroups>=1`（支持 `-DMX_GROUP_RANKS=32` 全组实验）。

### 2. 扫描结果（`-s2 top50 -replays 3`，mean µs）

| 配置 | mean |
|---|---:|
| **stage3b 默认（group-4, cost=ceil16, skew=3）** | **564.6** |
| skew 阈值 1 | 569.4 |
| cost=rows | 1104（回退） |
| cost=ceil16+4 | 611（回退） |
| `MX_GROUP_RANKS=8`（组 8，权重 2.75GB） | 598 |
| `MX_GROUP_RANKS=32`（全组，权重 10GB） | 948（权重体积致 cache/TLB 回退，s1 也大退） |
| 强制 win=64（runtime） | 578 |

### 3. 结论
- **group-4 为最优组规模**；组越大权重体积越大，远端/缓存代价压过均衡收益。
- 550µs 在当前「组内 4-rank 窗口重分配」框架下未达成：模拟显示组 4 最大负载仍为理想 2.2×，
  wall 由最忙 block 的**单个大窗（≤64 行）任务**决定；rank 级重分配对该单任务无效。
- 跨组均衡被**权重复制边界**阻断（helper 只能读本组 w1g/w2g；读远端/全量权重均回退）。
- 后续需**缩短单任务**：M 维切分（同窗多 block 分摊行）或 K 切（`R19_BLOCK_KSPLIT`）与 gs 组合。

## [261013_0024] agent/peerbuf/stage3c：**默认开启** 组内均衡（MX_GROUP_SCHED 默认 1，人类指示）

> 按人类指示把 `MX_GROUP_SCHED` 默认改为 **1**（`-DMX_GROUP_SCHED=0` 可关回旧基线）。
> 默认构建（无 KERNEL_CFLAGS）现即走组内均衡 + 远端 helper。

### 门禁 / 数据（默认构建，R=32）
- `check --smoke` **PASS 8/8**，7.4s。
- `-s2 top50` mean **568.5µs**（replays3）；旧默认（src 内 gs=0）≈695µs → **−18%**。
- `s1 TPR16/32/256` = **327/433/1611µs**（旧默认 315/417/1604，均衡 +0.5~3.9% 回退，已知）。
- `sizeof(rwk_data)=1543610624B`（w1g/w2g 32 专家；旧默认 637MB）。

### 说明
- 仅倾斜路由净收益；均衡路由有回退。默认开启为人类决定，后续需解决均衡回退。
- `-DMX_GROUP_SCHED=0` 行为与 f0abe9e 之前默认逐字节一致（同一 gs_cap=false 路径）。

## [261014_06_00] agent/gsfix：**修复 gs-on 模型 ATU**（commit `b82bbd5`）— gs-on 首次跑通且更快

> 根因：`torch/csrc/apis/layout.hpp::prod_smem`（`mega.hpp:259` 按 `window=8` 调用）
> 只给 ~8448B，而 `[MX_GROUP_SCHED]` kernel 动态 smem 需 **8516B**（R=32、kGroupRanks=4）
> → `s_prefix/s_gscount/s_ge/s_gw` 越界 → 垃圾 `ge`/owner → 远端 ATU。bench 走
> `host_layout.cuh` 的 40KB 故不复现。修复：`prod_smem` 取 `max(原公式, 40960)`；
> 加 `[GS-CAP]` 钳位；`MX_GS_SKEW_NUM` 默认改 **2**。`kMaxTPR`/`kMaxCombineRows` 曾试
> 512→768/1024（发现只是"推迟"ATU；真因确认后在 `[261014_07_00]` 回 512）。

模型：GLM-5.2-FP8 @ 10.0.82.71，EP=32，MTP=512，win48，CUDA graph decode，
16k/1k、bs128、prefix-cache99%、`bench_serving.sh decode`：

| 配置 | ATU | bench | Mean TPOT (ms) | Output (tok/s) |
|---|---|---:|---:|---:|
| base（gs off，win48） | 0 | 完成 | 30.62 | 4002 |
| gs-on（`b82bbd5`，kMaxTPR=768, SKEW=0） | 0 | 完成 | 26.78 | 4525 |
| **gs-on（`5dd7ced`，kMaxTPR=512, SKEW=2）** | **0** | **完成** | **26.33** | **4610** |

> 结论：gs-on 修复后**首次跑通**，TPOT **−14%**（30.62→26.33ms）、Output **+15%**
> （4002→4610 tok/s）；`kMaxTPR=512`（原始值）即足够，ATU 真因是 torch smem 溢出。
> `check --all` 762/762。

本地 `mybench`（R=32，均衡/倾斜，`us_mean`）：

| 配置 | s1 TPR16 | s1 TPR32 | s1 TPR256 | s2 top50 mean |
|---|---:|---:|---:|---:|
| SKEW=0（曾默认） | 411 | 530 | 1804 | 583.6 |
| **SKEW=2（当前默认）** | 379 | 457 | 1625 | 584.7 |
| SKEW=3 | 366 | 471 | 1639 | 586.4 |

> `s2 top50`（真实倾斜路由）与 SKEW_NUM 无关（2/3/0 都会启用 gs）；差异只在**均衡** s1：
> SKEW=0 差、2 与 3 相当且远优于 0。kMaxTPR 512 vs 768 对本地 bench 基本无影响。
> 热路径无调试代码（printf 全部 env/测试门控）。



## [261014_19_46] agent/glm52a/opt：[L2TUNE-49] L2 49..63 慢回退修复 — 均衡 TPR 49..63 全线抹平（−91..−102µs）；coverage nt=54/60 −93/−99µs；65case 38/65 提速（均值 −9.75µs）

> 根因：L2 分派链 [OPT-64] 门控 `count%(4*kM)==0` 在 count≤64 时仅放行 64，
> 49..63 掉进 `r19_l2_big<4,PF,3>` 慢回退（L1 侧 [L1TUNE-D] 已修复同构问题，
> L2 未镜像）。修复=去门控 + 函数内二分支：64 无掩码；49..63 掩码 MT4
> `r24_l2_big<4,KS,3,true,4>`（与 L1TUNE-D 同构）。src/compute_big.cuh 14+/6-，
> 无开关/宏。commit `34cdcb4` 之后的未提交修复，入提交见本节时间标签。

本地 mybench（`-R 32 -s 1`，W64，us_mean）：

| TPR | 48 | 49 | 54 | 60 | 64 | 66 |
|---|---:|---:|---:|---:|---:|---:|
| 修复前 | 1122.9 | 1302.3 | 1399.5 | 1499.5 | 1461.9 | 1253.5 |
| 修复后 | 1124.0 | 1211.1 | 1302.8 | 1399.7 | 1456.0 | 1255.3 |
| Δ | +1 | −91 | −97 | −100 | −6 | +2 |

coverage 21 点 harness（graph 50-in-1 rankavg，ms，`rd_cov_glm52a`→`rd_cov_l2fix`）：
**54: 1.3630→1.2704（sp 0.549→0.585）、60: 1.4756→1.3767（0.536→0.566）**；
其余 19 点 ±15µs 抖动。

65 定档 harness（`--routing-fixed-cases all`）：mega 逐点 Δ 均值 **−9.75µs** /
中位 −9.10；38/65 改善、8/65 小回退（+5.0..+12.5）；speedup≥1.0 = **64/65**
（853900 由 1.000→0.989 边界擦线），均值 **1.1761→1.1943**。
TK top50 mybench A/B（每产物 3 跑交替均值）：Δ 均值 **−14.35µs** / 中位 −12.37；
最大 −47.7（804900）/ −37.9（807100）/ −37.4（808800）。

门禁：`check --all` **762/762**（100.4s）、`check --smoke` 11/11；部署产物
e3485a54（旧 65bd0632 备份 `backup_pre_l2fix/`）。残余：TPR=49 仍高出 48↔64
弦线 ~66µs（MT3→MT4 16 行 tile 量化 + 单窗任务并行度），属结构差。
详录：README「coverage 均衡 21 点调研」8)。
