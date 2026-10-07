# test/76.real_decode — 真实 GLM-5.2 PD 分离 decode 专家分布数据集

- 源：`tmp/sources/PD分离_bs128_16k1k_专家分布.log`（真实推理 DG_TK_INFO 采样）。
- 生成：`gen_dataset.py` → `dataset.tk_info`（bench `--tk-info` 可直接消费）。
- 规模：1797 launch 中 346 个满足 bench 的 `total%(R*topk)==0`（其余 per-rank
  token 数 ragged），均 32 rank × 8 专家。
- 口径：`./bench --tk-info test/76.real_decode/dataset.tk_info --tk-all
  --ranks 32 --nodes 20 --replays 2`。

## 分布特征（真实 decode）
- **长尾占比 100%**（max_expert ≥ 3×median(非零)）；全局 max_expert median **209**、
  p90 314、max 398；非空专家 median 240/256。
- 每 launch per-rank tokens（=total/256）：3~16（远小于 tk 日志的固定 16）。
- 路由**逐 launch 不同**（rank0 抽样 91% 唯一）→ 每层 router 不同，`prologue_reuse`
  不适用（保持 False）。

## merge=0 vs merge=1（346 launch）
| 指标 | merge=0 | merge=1 | 变化 |
|---|---:|---:|---:|
| mean | 638.2 | 581.6 | **−8.9%** |
| median | 628.1 | 580.2 | −7.6% |
| p90 | 682.5 | 674.8 | −1.1% |
| max | 1511.3 | 1001.7 | **−33.7%** |

## 相位组成（344 launch，PHASE_DUMP，rank0）
| 相位 | mean | median |
|---|---:|---:|
| us（全链 mega+REDUCE） | 639.9 | 629.0 |
| mega | 406.6 | 356.9 |
| **REDUCE（独立 kernel）** | **233.3** | **260.7** |
| dispatch | (61.5) | 61.5 |
| L1 | 206.2 | 165.2 |
| L2tail | — | 121.8 |

- 真实 decode：**mega ≈ 64%，REDUCE ≈ 36%**；极端例（max_expert=398）
  REDUCE 占 **70%**（634µs）。
- 与 bench 的 tk 日志（REDUCE ~50~100µs）显著不同 → 真实 decode 是
  **REDUCE/COMBINE 长尾**主导。

## 结论 / 下一步
- merge=1 修复在真实分布上有效（−9% mean，极端 −34%）。
- 该场景最大剩余杠杆是 **REDUCE/COMBINE 的跨 rank straggler**（36%，极端 70%），
  而非 GEMM（A 已隐藏、B 到峰值）。下一步应针对 reduce 等待/合并优化。

## 复现
```bash
python3 test/76.real_decode/gen_dataset.py
./bench --tk-info test/76.real_decode/dataset.tk_info --tk-all --ranks 32 --nodes 20 --replays 2
```

## [261007] 深入诊断（目标：再优化 30%）

同会话消融（merge=1，346 launch）：
| 变体 | mean µs | 说明 |
|---|---:|---|
| rbase | 582.8 | 基线（merge=1） |
| 去 A（NOALOAD） | ~574 | A 装载≈9µs |
| 去 B（NOB） | ~549(merge0) | B≈88µs |
| 去 A+B | ≈549 | GEMM 操作数合计 ~14% |
| 空 REDUCE（保留 launch） | **330.7** | −43% |
| 去 REDUCE 等待（REDUCE_NOWAIT） | **337.4** | −42% |

结论：
- 链 = **最慢 rank 的 mega（mean 495.6µs）+ reduce work（~7µs）**；
  reduce 的**等待 246µs = 跨 rank 不均衡**（相关 corr(reduce, max_mega−本rank_mega)=0.90）。
- 均衡 TPR=1 的每 launch **地板 264µs**，且 L1+L2=207µs ≈ 读全部 8 本地专家权重
  （~301MB/rank ÷ ~1.45TB/s）→ **小 token decode 的固定成本是权重读**。
- 消融均无效：`__threadfence` 去除（+0）、PULL 远端↔本地（−6µs）、
  win_cap 48 已最优（16/24/32/64/96/128 均更差）、kNSlice 4 已最优（2/1 更差）。

瓶颈性质：真实 decode = 权重读地板 + 路由偏斜导致的跨 rank 长尾（reduce 等待）。
两者都不在 kernel 内部可轻易削减（权重必须读；偏斜由 router/EP 决定）。
