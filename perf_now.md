# 当前性能快照（perf_now.md）

> 记录当前 main 的 `-s 1`（均衡）与 `-s 2 top 50`（TK 实录不均衡）性能。
> 采集时间：`[261011_1441]`（自适应选窗 `[261011_1417]` + 尾块 MT 档 `[261011_1441]` 后）。
> 口径：`test/mybench`，R=32，nodes=20，replays=2，us_mean（跨 rank）。

## 0. 选窗策略（当前：自适应）

- `test/mega_common.cuh::choose_win_cap_counts(per_expert, max_tokens)` 调用
  `mxk::select_win(counts, E, cap_hint=max_tokens, win_min=32)`（`src/win_select.cuh`），
  **逐 launch 自适应选窗**：
  - 候选 `C = {32,48,64,96,128} ∩ [32, max_tokens]`；
  - 代价模型 `T̂(win) = waves(win) · t_task(win)`，`waves=ceil(W·kNSlice/kGrid)`、
    `W=Σ_e ceil(c_e/win)`；`t_task` 为逐候选标定表（Q8 定点，`docs/11`，捕捉 count>64 台阶）。
- `src/megakernel.cu` host 守卫：`win==slot` 或 `16 倍数且 32≤win≤slot`。
- `MT_FORCE_WIN=N`（mybench/check）可强制固定 win（调试）。
- 正确性：全量 `check --all`（自适应）**PASS 762/762**；另 win=32/48/64/96/128 各全量
  **762/762 PASS**（多窗/多波路径正确）。

## 1. `-s 1` 均衡（自适应，win_global=128）

```bash
scripts/run.sh mybench -R 32 -s 1 \
  -tprs 1,2,4,6,8,12,18,24,30,36,42,48,54,60,66,72,78,84,90,128,256 -nodes 20 -replays 2
```

| TPR | us_mean | TPR | us_mean |
|---:|---:|---:|---:|
| 1 | 303.6 | 54 | 636.5 |
| 2 | 307.7 | 60 | 645.6 |
| 4 | 318.1 | 66 | 727.0 |
| 6 | 322.1 | 72 | 737.5 |
| 8 | 328.4 | 78 | 734.9 |
| 12 | 337.7 | 84 | **853.0** |
| 18 | 458.1 | 90 | **847.4** |
| 24 | 485.1 | 128 | 984.4 |
| 30 | 481.5 | 256 | **1753.6** |
| 36 | 508.0 | | |
| 42 | 512.9 | | |
| 48 | 543.2 | | |

> 均衡下自适应选 `win_global=128`；相对旧默认 `win=512`，TPR=256 由 1857.7→**1753.6（−5.6%）**。
> `[261011_1441]` 尾块按精确活跃 tile 选 `MTAct`（补 2/3/4/6/7 档）：TPR 84/90 由 955→**853/847（−11%）**，
> 96/100/112 亦降；`count∈65..127` 原一律按 MT8（128 行）白算。

## 2. `-s 2 top 50` TK 实录（自适应）

```bash
scripts/run.sh mybench -R 32 -s 2 -top 50 -nodes 20 -replays 2
```

| 指标 | 值 |
|---|---:|
| sum us_mean（50 用例） | **40162** |
| mean | 803.2 |
| min | 661.6 |
| max | 979.0 |
| 选中 win 分布 | `48:39, 64:4, 128:7` |
| tokens / max_exp | 全 16 / ∈[168,505] |

> 相对旧默认 `win=512`（sum=112080），自适应 **−64%**；接近最优固定 `win=48`（39176）。

## 3. 备注

- 自适应为 **host 端逐 launch**（graph 捕获前定型）；若需 replay 内按负载自适应，仍需
  `docs/10` 的设备端选窗。
- `perf.md`「当前基线快照」为 `win=128` 口径，与本文大档可对齐。
- 采集耗时：`-s1` 21 档 ≈8s；`-s2` top50 ≈11~16s（>10s，属一次性记录而非门禁测试）。
