#!/usr/bin/env python3
"""从真实 GLM-5.2 PD 分离 decode 日志提取 TK_DUMP → bench TK_INFO 数据集。

源: tmp/sources/PD分离_bs128_16k1k_专家分布.log（[TK_DUMP] rank launch n=8 k : c0..c7）
输出: test/76.real_decode/dataset.tk_info

约束：bench `load_tk_all` 要求每 launch 的 total %(R*topk)==0（R=32,topk=8）；
真实日志 per-rank token 数不规则，故只保留可整除的 launch（346/1797）。
"""
import re
import sys
import collections

SRC = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/lchen1/data/projs/mini_mega/tmp/sources/PD分离_bs128_16k1k_专家分布.log"
OUT = sys.argv[2] if len(sys.argv) > 2 else \
    "/home/lchen1/data/projs/mini_mega/test/76.real_decode/dataset.tk_info"

txt = open(SRC, errors="ignore").read()
pat = re.compile(r"\s*rank=(\d+)\s+launch=(\d+)\s+n=(\d+)\s+k=(\d+)\s*:\s*([0-9 ]+)")
by = collections.defaultdict(dict)
for seg in txt.split("[TK_DUMP]")[1:]:
    m = pat.match(seg)
    if not m:
        continue
    c = [int(x) for x in m.group(5).split()]
    if len(c) == int(m.group(3)):
        by[int(m.group(2))][int(m.group(1))] = c

valid = []
for launch in sorted(by):
    d = by[launch]
    if len(d) != 32:
        continue
    total = sum(x for r in range(32) for x in d[r])
    if total > 0 and total % (32 * 8) == 0:
        valid.append(launch)

with open(OUT, "w") as f:
    f.write("# 由 %s 转换（真实 GLM5.2 decode 专家分布）\n" % SRC.split("/")[-1])
    for launch in valid:
        for rank in range(32):
            f.write("[TK_INFO] rank=%d launch=%d n=%d : %s\n"
                    % (rank, launch, 8, " ".join(map(str, by[launch][rank]))))
print("wrote %d launches -> %s" % (len(valid), OUT))
