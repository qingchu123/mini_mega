# test/75.fg_rwk_local_read — 同布局 FG vs RWK 本地读带宽微基准

- 日期：2026-10-07；机器：202（MetaX C600-UL ×32，32 SM/卡）
- 容器：`docker exec cl_test`（宿主 `/opt/maca` 与驱动 ISA 16.10 不匹配，
  bench 直接跑会 `libmcpti.so` 符号错误 / `xcore1610` 重编译失败；容器内正常）。
- 口径：每个 size 一次 kernel 读完整缓冲；预热 3；FG/RWK **交错各 15 轮取 min**；
  event 夹住；带宽 = size / 时间。两缓冲同填充、reduce 校验一致（PASS）。
- 布局：`contig16` = uint4 连续读（raw 上限）；`afrag` = 复刻 `cm_load_a`
  的 native-fragment 8B/lane、16 行×32B tile（行主序 pool 的实际读法）。

## 结果（RWK 相对 FG；3 次独立运行）

| size | 布局 | FG GB/s | RWK GB/s | RWK vs FG |
|---:|---|---:|---:|---:|
| 8 MB | contig16 | 273~285 | 275~285 | −0.9 ~ +0.9% |
| 8 MB | afrag | 107~109 | 107~109 | −0.3 ~ +0.3% |
| 32 MB | contig16 | 441~447 | 443~447 | 0.0 ~ +0.3% |
| 32 MB | afrag | 127 | 127~129 | 0.0 ~ +1.5% |
| 128 MB | contig16 | 520~522 | 520~522 | −0.2 ~ 0.0% |
| 128 MB | afrag | 423~428 | 445~454 | **+4.8 ~ +7.1%** |
| 512 MB | contig16 | 1500~1645 | 1495~1636 | 噪声大（−9 ~ +7%） |
| 512 MB | afrag | 420~426 | 441~443 | **+4.0 ~ +5.2%** |

## 结论

1. **L2 驻留（≤32MB）：FG 与 RWK 本地读无差别（<1%，噪声内）。**
2. **超 L2（≥128MB）走真实 A 散射布局（afrag）：RWK 稳定快 ~5%**（FG/RWK 交错
   测量，contig16 同规模无此差 → 非测量偏置，是访问模式相关）。
3. **主成本是布局不是显存类型**：afrag（行主序 16 行散射）仅
   ~420~450 GB/s，而连续读 ~1400~1650 GB/s（3~4×）。即 L1 A 的瓶颈在
   16-line 散射装载，与 FG/RWK 无关。

## 对「L1 A 放 RWK」的判断

- dispatch 为 pull（远端读 FG 的 x、写本 rank pool，compute_big.cuh:1137/1152），
  pool 是**本地写**，正确性上不强制 FG → 单独放 RWK 可行。
- 但真实收益仅 ~5%（且只在 HBM 绑定、A 走散射布局的大 count 场景）；A 装载
  占 L1 的一部分，端到端预计 ~1% 量级，低于会话波动。
- 更值得做的是**布局**（native-fragment 落盘，R23 poolfrag）——afrag↔contig 的
  3~4× 差距远大于 5%。

## 复现

```bash
bash test/75.fg_rwk_local_read/build.sh            # 宿主（需 CUCC_TARGETS）
# 本机宿主工具链与驱动 ISA 不匹配，实际用容器：
docker exec cl_test bash -lc 'cd <repo>/test/75.fg_rwk_local_read && \
  $MACA_PATH/tools/cu-bridge/bin/cucc -O3 -std=c++17 \
  -I$MACA_PATH/tools/cu-bridge/include -I/opt/maca/include -o /tmp/fg_rwk_read fg_rwk_read.cu'
docker exec cl_test bash -lc 'MACA_VISIBLE_DEVICES=0 /tmp/fg_rwk_read'
```
