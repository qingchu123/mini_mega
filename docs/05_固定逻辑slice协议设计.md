# 05 固定逻辑 slice 协议（去除 nslice_eff 跨 rank 依赖）设计草稿

状态：草稿，未实现。本文只描述第 1 步（协议层），不涉及 win 的设备端选择（第 2、3 步另文）。

## 1. 背景与动机

当前 combine 的完成标记与 REDUCE 等待都依赖 `nslice_eff`：

- `compute.cuh:957` 由 `merge`（[R35] 波量化粗化）推出 `nslice_eff = kNSlice / merge`，
  并在 `compute.cuh:963` 写入 `bar[...][kBarNsliceEffIdx]`（`common.cuh:40`，值 1021）。
- `combine.cuh:534-540`：REDUCE 读取该广播槽，等待 `kTopk × ns_eff` 个 `combine_sums` 标记。
- `combine.cuh:429`：列段划分 `cols_per_slice = kHidden / c.nslice_eff`。

`nslice_eff` 是**本 rank** 的量（由本 rank 的 `s_cum`/`merge` 决定），但它被**源 rank** 的
REDUCE 当作等待数使用。源 rank 的 REDUCE 能否正确放行，取决于它与目标 rank 的 `nslice_eff`
是否一致。这是 win/merge 必须全 rank 一致的直接来源（见 perf.md 的 v5 与 [FIX-NSE] 注释）。

目标：REDUCE 的等待集合与列段划分只依赖与 rank 无关的常量 `kNSlice`，使 `merge` /
`win` 可以在每个 rank 上独立决定。

## 2. 当前协议（as-is）

### 2.1 目标端（发送 COMBINE 的 rank）
- 每个物理任务 `c.slice ∈ [0, nslice_eff)`，覆盖逻辑 slice `[slice*merge, slice*merge+merge)`。
- 推送列段：`c0s = slice * cols_per_slice`，`cols_per_slice = kHidden / nslice_eff`。
- 完成标记：对覆盖的每个逻辑 slice 写 `combine_sums[cycle][k][token][logical]=epoch`
  （`combine.cuh:402-412`、`combine.cuh:466-478`，已是 `merge` 个标记写满）。

### 2.2 源端（REDUCE）
- 等待 `tid < kTopk * ns_eff`，`k = tid / ns_eff`，`s2 = tid % ns_eff`，
  轮询 `combine_sums[cyc][k][token][s2]` 直到 `>= epoch`（`combine.cuh:529-552`）。
- `ns_eff` 来自 `bar[cyc][kBarNsliceEffIdx]`，由 dispatch 的 block0 写入。

### 2.3 问题
- 目标端列段按**本地** `nslice_eff` 切；源端等待按**本地** `ns_eff` 切。二者若不同，
  源端会等一个"不存在"的标记（漏槽超时）或少等一个标记（读到未写完的列段）。
- 当前代码能工作，只因为 win/merge 全 rank 一致。

## 3. 目标协议（to-be）

### 3.1 不变量
- **I1**：标记粒度固定为逻辑 slice，数量 `kNSlice`。标记 `combine_sums[c][k][token][L]`
  表示"逻辑 slice L 的列段已由目标端写完"。
- **I2**：列段划分固定：`cols_per_logical = kHidden / kNSlice`，逻辑 slice `L` 覆盖列
  `[L*cols_per_logical, (L+1)*cols_per_logical)`。该划分与 rank、win、merge 无关。
- **I3**：REDUCE 等待 `kTopk × kNSlice` 个标记（全部逻辑 slice）。不读任何广播值。
- **I4**：物理任务 `slice_phys` 覆盖逻辑范围 `[slice_phys*merge, slice_phys*merge+merge)`，
  对该范围内每个 L 推送列段 `[L*cpl, (L+1)*cpl)` 并写标记 `L`。`merge` 只影响本 rank 的任务
  切分，不影响协议。

### 3.2 与 merge 的关系
- `merge` 继续存在（仍是本地的任务粗化，用于控制物理任务数 ≤ grid）。
- 粗化后一个物理任务推送多个逻辑列段。实现方式：在 `combine_push_pair` 内对 `merge` 个
  逻辑 L 顺序推送（当前 `kNSlice % merge == 0` 已保证整除）。
- 代价：粗化任务的推送量不变，只是被拆成 `merge` 段；标记写入量不变（本来就是 `merge` 个）。

### 3.3 与 kNSlice 的关系
- `kNSlice` 仍是编译期常量（`KNSLICE_OVERRIDE`，默认 4）。
- 约束：`kHidden % kNSlice == 0` 且 `(kHidden / kNSlice) % 8 == 0`（每 lane 8 列）。
  kHidden=6144，kNSlice ∈ {1,2,3,4,6,8,...} 满足；kNSlice=3 时 cpl=2048，8 整除，合法。

## 4. 需要改动的点（逐文件）

| 文件 | 位置 | 改动 |
|---|---|---|
| `src/common.cuh` | 36-40 | 删除 `kBarNsliceEffIdx` 注释与常量（或保留常量但标记为 unused，便于回退） |
| `src/compute.cuh` | 961-964 | 删除 `nslice_eff` 广播写入 |
| `src/combine.cuh` | 153-156 | `nslice_eff` 字段保留为本地几何量（仍用于任务到列段的映射），但不再用于协议 |
| `src/combine.cuh` | 429-431 | `cols_per_slice` 改用 `kHidden / kNSlice`；`c0s` 由逻辑 slice 推导（见 3.2） |
| `src/combine.cuh` | 529-552 | REDUCE 等待改为 `tid < kTopk * kNSlice`，`s2 = tid % kNSlice`，删除 `ns_eff` 读取 |
| `src/combine.cuh` | 402-412、466-478 | 标记写满逻辑范围（已如此，核对即可） |
| `src/combine.cuh` | 427-470 | 列段推送按逻辑 L 循环，覆盖本物理任务的 `merge` 个 L |
| `test/check.cu` | 586 | 诊断打印 `kBarNsliceEffIdx` 的部分删除或改为打印 `kNSlice` |

不改动：`compute_big.cuh` 中 `slice_flags/l1_prog` 的 `nslice_eff` 用法（这些是**本 rank 内**
的 L1/L2 完成标记，不跨 rank，见 4.1）。

### 4.1 本 rank 内的 nslice_eff 保留范围
`compute_big.cuh` 的 `slice_flags`、`l1_prog`、`ring` 门（`compute.cuh:1083-1086`）只在本 rank
内用于 L1→L2 的依赖，不跨 rank。它们继续使用本地 `nslice_eff`，不在本次改动范围。

需要核实：`l2_done`、`task_ready` 是否跨 rank 读取。若跨 rank 读取，则需纳入协议。
（当前代码注释显示 l2_done 是本 rank 的 task 计数，待第 2 步确认。）

## 5. 正确性论证

- **标记完备性**：每个逻辑 L ∈ [0, kNSlice) 恰由一个物理任务覆盖（物理任务范围划分不重叠、
  覆盖全集，由 `total_tasks * merge == raw_tasks` 与 `kNSlice % merge == 0` 保证）。
  因此每个 `(k, token, L)` 标记恰被写一次，REDUCE 等待 `kTopk × kNSlice` 个标记可满足。
- **列段完备性**：`L` 对应列段固定，REDUCE 读取列段 L 时，等待的标记 L 对应同一列段。
- **跨 rank 无关性**：源端等待集合、标记编号、列段划分均只依赖 `kNSlice`、`kHidden`、`kTopk`，
  与目标端的 win/merge 无关。目标端不论怎么切任务，只要标记覆盖全集即可。
- **epoch 语义**：标记值仍为 epoch（单调），REDUCE 以 `>= epoch` 放行，不受任务数变化影响。

## 6. 风险

1. **标记写入竞争**：粗化任务（merge>1）中，同一物理任务覆盖的逻辑 L 之间无依赖，
   顺序写入即可；但需确认 `combine_sums` 的写入不会被同一 warp 的其他 lane 覆盖（已有
   `if (lane == 0)` 逐 L 写的模式，沿用）。
2. **REDUCE 线程数**：`kTopk × kNSlice` 可能超过 `blockDim`（256）。kTopk=8，kNSlice=4 时为 32，
   可以；kNSlice=16 时为 128，仍可；但 `tid` 循环需保证 `kTopk*kNSlice <= blockDim.x`，
   需在实现中加静态断言。
3. **merge=1 多波的 GEN 代际旗**：`slice_flags/l1_prog` 的代际逻辑（perf.md v-merge1 节）只在本
   rank 内，本次不改动，但需要在 merge=1 用例上回归。
4. **性能**：标记等待线程数从 `kTopk × ns_eff` 变为 `kTopk × kNSlice`。merge>1 时，原来 REDUCE
   只等 `ns_eff` 个标记，改后等 `kNSlice` 个；每个 REDUCE block 的轮询量增加至 `merge` 倍。
   预期影响小（标记已被写满、轮询即刻返回），需用 perf 验证。
5. **回退**：保留 `kBarNsliceEffIdx` 常量与广播写入（但 REDUCE 不再读取）作为回退开关，
   待验证通过后再删除。

## 7. 验证计划

按固定顺序，每步通过后再进行下一步：

1. **编译**：`./scripts/build.sh so check mybench` 通过。
2. **smoke**：`TIMEOUT=300 ./scripts/run.sh check --smoke`，全部 PASS。
3. **子集**：`check --all --only "$(cat .agents/work/subset10.csv)" --tprs 64`，12/12 PASS。
4. **top50 列表**：`check --all --only "$(cat .agents/work/top50_list.csv)" --tprs 64`，52/52 PASS。
   （注意：当前 check 耗时约 12 s，超过项目 10 s 要求，需在验证时记录，不以此阻塞。）
5. **merge=1 用例**：`MX_MERGE1=1` 下 TK 长尾 launch 的 check，确认 GEN 代际路径不回归。
6. **性能**：`mybench -s 1`（TPR=1,4,16,32,33,64,128,256,512）与 `mybench -s 2 -top 50`，
   与 perf.md 新基准（`[261009_12_50]` 节）逐项对比。判据：均衡 TPR 相对基准 ±3% 内；
   `-s 2 -top 50` 不劣于基准。
7. **节点数复测**：`TIMEOUT=900 ./scripts/run.sh mybench -s 1`，若 GPU 队列超时（环境问题，
   见 perf.md 记录），重跑而非判定为回归。

## 8. 不在本次范围

- 设备端 win 选择（第 2 步）。
- 删除 `_select_adaptive_cfg`（第 3 步）。
- rank 不一致 win 的专项测试（第 4 步）。
- 守卫的放宽（已在工作区，未提交，本次不改）。

## 9. 待确认事项（实现前需要你决定）

1. **是否保留 `kBarNsliceEffIdx` 作为回退开关**：建议保留到验证通过，再删除。
2. **check 的 `kBarNsliceEffIdx` 诊断打印**（`test/check.cu:586`）：改为打印 `kNSlice`，还是直接删除该诊断？
3. **验证顺序**：建议先在当前 win 一致的前提下完成第 1 步全部验证，再进入第 2 步。
4. **提交粒度**：建议第 1 步单独一个提交，标签格式 `[时间] refactor(combine): ...`，性能数据写入 perf.md。

## 10. 实现记录（[261009_13_13] 第 1 步）

- 已删除 `kBarNsliceEffIdx` 广播与常量；REDUCE 等待改为 `kTopk×kNSlice`（static_assert ≤256）。
- `combine_push_pair` 改为逐段推送本物理任务覆盖的 `merge` 个逻辑列段（`cpl=kHidden/kNSlice`）。
  与旧写法（`cols_per_slice=kHidden/nslice_eff`）数据范围数学等价，旧写法并无覆盖缺失。
- 探针确认：merge=2 时 `fuse_on=0`，走非融合 `combine_push_pair`（510 次/2 用例）。
- 验证：smoke 5/5、子集 12/12、top50 52/52（默认、win=48/64/128）全部 PASS。
- 性能：`-s 1` 各档 ±1.2%；`-s 2 -top 50` 由 3274 降至 3149 µs（两次复测一致）。
  等价写法不应改变数据量，该差异原因未确认，记录为待查，不作为已确认的优化。

## 11. 不一致 win/merge 实验（rank 奇偶不同 win）

方法：临时在 check 中按 rank 奇偶设置不同 win（`MT_FORCE_WIN` 偶数 rank、`MT_FORCE_WIN_ODD` 奇数 rank），
布局仍按各 rank 最大 win 统一分配。实验补丁未提交（已还原）。

| 偶 rank win | 奇 rank win | 用例 | 结果 |
|---:|---:|---|---|
| 64 | 128 | 子集 12 | 12/12 PASS |
| 64 | 128 | top50 | 52/52 PASS |
| 48 | 96 | top50 | 52/52 PASS |

探针（64/128 子集）确认 raw 任务数对应 merge=1 与 merge=2 并存，即 rank 间 merge 确实不一致。

结论：第 1 步之后，rank 间 win（因此 merge）不一致不影响功能。
范围限制：仅 check harness（单布局、统一 win_global）；torch 真实路径、多机、非 16 倍数 win 未测。
