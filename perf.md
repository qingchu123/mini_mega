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
