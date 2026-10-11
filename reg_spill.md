# megakernel 寄存器 spill 问题分析

> 仅分析，不含代码改动。基线：`main` @ `20820cc`（[261011_1257]）。
> 结论一句话：生产核 `mega_kernel<true>` 每线程寄存器 **顶格 256**、栈帧 local ≈ **2.5KB**，
> 溢出（spill to local）真实存在；热点集中在 **大任务融合体 `r19_big_task`** 与其内联的
> GEMM 主体，根因是 **累加器 `acc[kBatch][kMT][kNT]` + A/B 寄存器预取环** 在 256 寄存器
> 预算内放不下，被编译器挤到 local memory。

---

## 1. 结论摘要

1. **溢出定位（静态反汇编，当前提交）**：
   - `r19_big_task<Both>`：栈帧 **1344 B**、栈最大偏移 1328 B、栈 ld/st **905 次**、52851 条指令
     → **明确的溢出点**（count>16 的大任务融合体）。
   - `r24_l2_big<8,8,2,true,8,2>`：栈帧 672 B、栈 ld/st 115 次；`<...false...>` 320 B。
   - `r24_l1_big` 全部实例与 `r24_l2_big<8,8,*>` 峰寄存器号到 **255（顶格）**，但多为 0 栈帧
     （靠寄存器硬扛，临界）。
   - `r19_small_task`（count≤16 热路径）：峰值 247 号寄存器、**0 栈帧 → 未溢出**（设计隔离生效）。
   - `mega_kernel<true>` 主函数自身：峰值 185 号寄存器、0 栈帧；溢出发生在它调用的 noinline 任务体里。
2. **运行时权威读数**（`MT_DUMP_RES=1` → `mcFuncGetAttributes`，见 `perf.md`）：
   `full` = **regs=256（顶格）、local≈2544 B**；L1/L2 拆分实验下 `L1full` 1680 B、`L2full` 1984 B。
3. **根因**：不是某个"临时变量多写了一个"，而是**结构性**——单个 GEMM body 的
   `acc` 就吃掉 96~128 个寄存器，再叠加预取环、地址/边界、pull 与 COMBINE 内联体，
   在 256 上限下必然溢出。

---

## 2. 分析方法与数据来源

| 来源 | 手段 | 用途 |
|---|---|---|
| `perf.md` (640, 664) | `MT_DUMP_RES=1` → `mcFuncGetAttributes` | 运行期权威 regs/local |
| `src/megakernel.cu:239 dump_kernel_resources()` | 打印 `[RES] regs/local/smem` | 触发上述读数 |
| `mxobjdump-open` + `llvm-objdump-open` | 从 `build/obj_so/kernel_compute.o` 提取 ELF 并反汇编 | 逐函数静态统计 |
| `.agents/work/agent-regspill/tmp/frame.py` | 解析反汇编：`sadd_co_u32 s32,imm,s32`(栈帧)、`ldp/stp _b32 s33,off`(栈访存)、最大 `rN` | 峰值寄存器号 / 栈帧 / 溢出载存次数 |

静态口径说明：MACA 里 `s32/s33` 分别是 **sp / 帧指针**；函数入口 `sadd_co_u32 s32, N, s32`
即分配 N 字节局部栈；`ldp/stp s33, off` 是对局部栈的读写，即 **spill 载/存**。
"峰值寄存器号"= 该函数中出现过的最大 `rN`，255 表示用满了全部 256 个寄存器。

### 2.1 逐函数静态统计（当前 `kernel_compute.o`，commit `20820cc`）

| 函数（模板实参） | 峰值寄存器 | 栈帧 B | 栈最大偏移 B | 栈 ld/st | 指令数 |
|---|---:|---:|---:|---:|---:|
| `r19_big_task<Both>` | **255** | **1344** | 1328 | **905** | 52851 |
| `r24_l2_big<8,8,2,true,8,2>`（L2 尾块 MT8） | 255 | **672** | 648 | 115 | 4805 |
| `r24_l2_big<8,8,2,false,8,2>`（L2 全块 MT8） | 255 | 320 | 304 | 54 | 4356 |
| `r24_l2_big<8,8,2,true,5,2>` | 255 | 192 | 184 | 32 | 3119 |
| `r24_l2_big<8,8,2,true,8,2>` 之外的 `r24_l1_big<*>` | 255 | 0 | 0 | 0 | 3~4.5k |
| `r24_l2_big<8,8,1,...,8,2>` | 255 | 0 | 0 | 0 | 4156 |
| `r19_small_task<Both>` | 247 | **0** | 0 | 0 | 19792 |
| `mega_kernel<true,false,Both>` | 185 | 0 | — | — | 3383 |

> 观察：`r19_big_task` 与 `r24_l2_big<8,8,2>` 是**唯二带真实栈帧**的（除小 L2 档的 16 B 占位）。
> 说明 spill 主要发生在"大 count / MT8"路径；小任务路径被 `__noinline__` 隔离后完全不落栈。

---

## 3. 寄存器预算：为什么 256 一定会溢出

- 生产核 `__launch_bounds__(512)`（`src/kernel.cuh:35`）：每 block 512 线程，编译器把每线程
  寄存器压到体系上限 **256**；这就是 `regs=256 顶格` 的来源，**不可能再往上加**。
- `__launch_bounds__` 同时意味着 2 个 block/SM 的占用约束；即便 `r19_small_task` 靠函数级
  `__noinline__` 隔离，只要**任一在链上的函数**要 ≥256 个活跃值，就必然 spill。
- 关键量级：一个 `R19Acc`/`Fp8Accumulator` = `__NATIVE_VECTOR__(4,float)` = **4 个 32 位寄存器**。
  单个 GEMM body 的 `acc[kBatch][kMT][kNT]` 就达：
  - `r24_l1_big<8,8,2>` → `2*8*2*4` = **128**
  - `r19_l1_big<4,2,4>` → `4*4*2*4` = **128**
  - `r24_l2_big<8,8,2>` → `2*8*2*4` = **128**
  128 已占一半预算，剩下一半要放 A/B 预取环、地址、指针、循环与 pull/tail 内联体 → 溢出。

---

## 4. 寄存器占用最多的代码（重点）

### 4.1 GEMM 累加器 `acc[kBatch][kMT][kNT]`（第一大户）

定义处：`r19_l1_big`（`src/compute_big.cuh:384`）、`r24_l1_big`（`:602`）、
`r24_l2_big`（`:890`）、`l1_gemm_swiglu_body_paired`（`src/compute.cuh:420`）、
`l2_gemm_body`（`src/compute.cuh:597`）。

`acc` 占用 = `kBatch * kMT * kNT * 4` floats（`kNT`=`kNTilesPerWarp`=2 恒定）。
编译进来的实例（取自符号表）：

| GEMM 实例 | acc（寄存器 float 数） | 用途 |
|---|---:|---|
| `r24_l1_big<8,8,2,...>` | **128** | L1 大档（>64 全块） |
| `r24_l2_big<8,8,2,...>` | **128** | L2 大档（>64） |
| `r19_l1_big<4,2,4,...>` | **128** | L1 17..32 档 / 兜底 |
| `r19_l1_big<3,2,4,...>` | 96 | L1 33..48 档 |
| `r24_l1_big<3,16,4,...>` | 96 | L1 33..48 async |
| `r24_l1_big<4,8,2,...>` / `r24_l2_big<4,8,3,...>` | 64 / 48 | 中档 / L2 49..64 |
| `l2_gemm_body<...,kBatch=6>` | 48 | L2 小档单批 |
| `r24_*_big<1,16,2,...>` | 16 | 尾块 tiny |
| `r19_l1_big<1,2,4/2,...>` | 32/16 | ≤16 热路径 |

> `acc` 是**跨整个 K 循环常驻**的（`k_iters=H/32=192` 步），编译器无法把它赶走，
> 只能溢出别的活跃值；当 acc=128 时，光是它就逼近预算上限。

### 4.2 A/B 寄存器预取环 `a_pipe` / `b_pair`（第二大户）

- `a_pipe[kPF][kMT]`（`Fp8MmaFragment=R19Frag=2` 寄存器）→ `2*kPF*kMT`
- `b_pair[kPF][kBatch]`（`R19Frag2=4` 寄存器）→ `4*kPF*kBatch`
- 位置：`r19_l1_big`（`compute_big.cuh:371-372`，`a_pipe[2][8]+b_pair[2][2]`），
  `l1_gemm_swiglu_body_paired`（`compute.cuh:397-399`），`l2_gemm_body`（`:581-582`），
  `r24_l1_big/l2_big` 的 `b_pair[kBPF][kBatch]`（`:594`/`:882`）与每 stage 的 `a_pipe[kMT]`（`:623`）。
- 设计文档明确记录：**加深 B 预取会直接溢出，回报为负**（`perf.md:261`；
  `perf.md:335` "加大 L1PF/L2PF/BPF 均回退"）。
- 注释 `compute.cuh:428` 强调 ring 必须 `#pragma unroll` 成编译期常量，否则
  "ring 退化 local memory"——这正是 spill 的另一种形态。

### 4.3 SwiGLU / 量化 epilogue 的临时量

- `R19Acc sw`（4 寄存器）与同时读 `acc[qa][m][t]`/`acc[qb][m][t]`（`compute_big.cuh:661-667`），
  同一时刻要保住 gate/up 两组累加器 + 打包临时（`__builtin_mxc_cvt_pk4_f32tof8_cfg`）。
- `r24_l2_big` 的 BF16 输出打包 `r19_pack_f2bf16` 与 `uint4` 装载。

### 4.4 地址 / 索引 / 边界计算

- 大量 64 位偏移：`MegaArgs` 的 `x_off/weights_off/pool_off/w_off/meta_off/prefix_off/
  combine_rows_off`、`S.*`（`src/compute.cuh:149,704-706`）；每个 `int64` 占 **2 个寄存器**。
- 递推指针：`w_e`、`a_c`、`f_c`、`pool_w`、`w2_e`、`ffn_slot`、`out_slot`、`fbase`、`ffn2`、
  `f2`，以及 `nbase[kBatch]`（`:367`/`:588`/`:876`）、`mt_total`、`m_abs0`、掩码 `row<cc`。
- `r24_issue_a`（`:522`）里的 `f/j/m/p/off/mb` 等循环展开索引同样占寄存器。

### 4.5 大结构体 `R19Sched` / `CombineCtx` / `CombineFuseCtx`

- `R19Sched`（`src/compute_big.cuh:272-312`）约 **30 个字段**（含多个 int64 指针/偏移）。
  按 `const&` 传（好），但函数体内 `cm_build_ctx(S)`、逐字段使用会把活值拉进寄存器。
- `CombineCtx`/`CombineFuseCtx`（`src/combine.cuh:141,173`）在 `r19_big_task` 内**内联构造**，
  尾段 `r19_tail_common` / `combine_push_chunk_rows` 也是**内联**，与 L1/L2 的地址空间叠加。

### 4.6 COMBINE 融合推送（内联在任务体里）

- `combine_push_chunk_rows`（`src/combine.cuh:214`）读 staging 的 `weights`（`w0..w3`）、
  `meta[stag][3]`，做 `uint4` 装载与 `combine_push_one_row`（每行多次远端写+地址运算）。
- REDUCE 的 `reduce_body`（`:482`）另有 `float w8[kTopk]`（8 寄存器）与 `float acc[8]`。
- 这些与 L2 的 `fuse.on` 分支（`compute_big.cuh:1570`）叠加，是 `r24_l2_big` 尾块实例
  出现 672 B 栈帧的直接来源之一。

### 4.7 前序 / PULL 内联体

- `dispatch_pull_body`（`src/compute.cuh:695`）被内联进 `mega_kernel`，携带 `kEPR*(R+1)`
  前缀 smem 指针、`s_count/s_cum/s_prefix`、多个 `bar_flags + *_idx`、
  `atomicAdd/reinterpret_cast` 计数与 `int64` 偏移，是一个"寄存器搬运工"。
- `r19_pull_flags`（行均分 K 片 pull + staging weights/meta）在任务体里同样内联，
  逐行地址运算 + 非法行零填充 + 分片旗，活跃值多。

### 4.8 诊断 / 编译开关遗留

- `pt`（`pre_ticks`）、`ctr`（`MX_DBG_COUNT`）、`MYBENCH_PHASE`、`R24_DIAG_PHASE` 等分支，
  虽然默认关闭，但 `pt`/`ctr` 指针与条件仍参与寄存器分配（`kernel.cuh:36-53`、
  `compute_big.cuh:1680-1684`）。
- 文档 `docs/06:128` 已列出待清理的 `R30_DIAG_*`/`CM_DIAG_*`/`R24_DIAG_*` 等宏。

---

## 5. 结构性问题：forceinline vs noinline 的取舍

| 函数 | 属性 | 后果 |
|---|---|---|
| `r19_l1_big`（`compute_big.cuh:346`） | `__forceinline__` | 内联进 `r19_big_task`；多个档位（acc 32/48/96/128）的模板实例**同函数**，编译器取并集峰值 → `r19_big_task` 1344 B 栈 |
| `l2_gemm_body` / `l1_gemm_swiglu_body_paired`（`compute.cuh`） | `__forceinline__` | 同上，叠加进任务体 |
| `r24_l1_big` / `r24_l2_big`（`compute_big.cuh:565/835`） | `R19_GEMM_ATTR=__noinline__`（`common.cuh:41`） | 独立寄存器/栈；**实测有效**：把重档从任务体摘出，TPR=16 回 ~24µs（`compute.cuh:688`、`docs/13:79`） |
| `r19_small_task` / `r19_big_task` | `__noinline__`（`:131/135`） | 大小任务互不污染；small 实测 0 栈帧 |

历史佐证（都在 `perf.md`）：
- 集14：`M4 单实例` 让整 kernel 顶格 256 + 栈 2032 B，污染全部路径；改 M1 分块后栈 176 B，
  graph 链 **−18~26%**（`compute.cuh:71-77`）。→ 证明 spill 会拖慢**非本档**路径。
- v8/v9：用编译期 `kCap` 删除不可达的 MT8/MT4 大档实例，前 50 launch **−3.1% / −4.0%**
  （`perf.md:308-358`）；`[OPT-KCAP-REMOVE]` 后该职责交给 `__noinline__` 隔离。
- L1/L2 拆 kernel：`local` 降到 1680/1984 B，但 **regs 仍 256、仍溢出**，且端到端 **净回退
  +7~9%**（`perf.md:642-670`）。→ 拆分不解决溢出本身，只是换个位置。

---

## 6. 为什么溢出集中在 big 路径，而 small 路径没有

- `count<=kM(16)` → `r19_small_task`：L1 走最小档 `r19_l1_big<1,2,4>`（acc **32**），
  L2 走 `l2_gemm_body<...,2/3>`（acc ≤48）。峰值 247 寄存器，**无需落栈**。
- `count>16` → `r19_big_task`：为覆盖 17..512 全部档，函数内**同时实例化**了
  `r19_l1_big<4,2,4>`(128)、`<3,...>`、`r19_l2_big<4,...>` 及 `l2_gemm_body<...,2,6>`、
  `l2_gemm_body<...,3,3>` 等；再加上 30 字段 `R19Sched`、pull、combine tail 内联，
  峰值 255 + 1344 B 栈。
- `__noinline__` 的 `r24_*` 虽然独立，但**自身**单个实例峰值也到 255，MT8 类仍溢出
  （`r24_l2_big<8,8,2,*>` 320~672 B 栈）——因为 128 acc + B ring + COMBINE 推送在同一函数。

---

## 7. 影响量化（来自既有记录）

| 证据 | 内容 |
|---|---|
| 集14（`compute.cuh:71-77`） | 单实例顶格 256 + 栈 2032 B → 同会话 graph 链 −18~26%；栈回到 176 B |
| v8/v9（`perf.md:308-358`） | 删重档实例：前 50 launch −3.1% / −4.0% |
| `perf.md:640` | `regs=256、local≈2.5KB`；"降 acc 档位/预取对 regs/local 基本无影响；`__noinline__` 拆分反而增大 local" |
| `perf.md:664-670` | 拆 kernel：local 降但 regs 仍 256、仍溢出，端到端净回退 +7~9% |
| `docs/12:181-201` | MT16 受 acc=128 顶格约束只能 kBatch=1 两遍，实测 1.6~2.3× 回退 |

> 注意：spill 的代价不只是"多几条 LDL/STL"，而是占用 L1/shared 带宽、拉长依赖、
> 并**外溢到同函数其它档位**（集14 已证伪"只影响本档"的直觉）。

---

## 8. 潜在方向（仅为分析，未实现、未验证）

按"收益/风险"排序（均需端到端 A/B，隔离微基准不可信，见 `docs/12:167-177`）：

1. **压缩 acc 峰值**：`r19_big_task` 内让不同档的 acc 生命周期完全不相交；或把
   `r19_l1_big` 也改 `__noinline__`（现为 forceinline），让每档独立 256 预算
   ——本仓库已证明该手法对 `r24_*` 有效（`common.cuh:38-42`），但 `perf.md:640` 提示
   `__noinline__` 会增加 local，需实测权衡。
2. **降低并发 acc**：`kBatch`/`kMT` 折中（acc ∝ kBatch·kMT）。已知 MT8 在 ≤256 最省权重但
   acc 顶格；MT16 已被否决。可探索 MT8/kBatch=1（acc=64）+ 两次 K 循环的中间档。
3. **把 COMBINE 推送移出 GEMM 函数**：`combine_push_chunk_rows` 内联进 `r24_l2_big`
   是尾块 672 B 栈的嫌疑之一；改为 `__noinline__` 独立帧，让 L2 主循环少背寄存器。
4. **地址/索引降宽**：能 32 位的偏移不用 int64；`nbase[]`/`mt_total` 等循环不变量提到
   最小活跃窗口。
5. **清理诊断宏**（`R24_DIAG_*` 等）与 `pt`/`ctr` 分支，减少无谓活跃值。
6. **接受 spill、优化访问**：若 acc 无法再降，则让 spill 命中 L1（当前 local 已走 L1 cache），
   并对齐 16B 让 LDL/STL 合并——属于"止损"而非"消除"。

> 反向教训（避免重复踩坑）：拆 kernel（L1/L2 split）、加深预取（L1PF/L2PF/BPF）、
> MT16/kBatch=1、全档 `R19_L1_MT=4` 均已实测回退或只在单场景成立（`perf.md:329-335`、
> `docs/12:179-201`）。

---

## 9. 复现命令

```bash
# 运行期权威读数（需 GPU）
MT_DUMP_RES=1 scripts/run.sh mybench -R 32 -s 1 -tprs 16 -nodes 1 -replays 1

# 静态反汇编统计（无需 GPU）
scripts/build.sh so                       # 生成 build/obj_so/kernel_compute.o
B=/opt/maca-20260926/mxgpu_llvm/bin/mxobjdump-open
$B --extract-elf --target-kind=maca-mxc-metax-macahca--xcore1610 \
   build/obj_so/kernel_compute.o
/opt/maca-20260926/mxgpu_llvm/bin/llvm-objdump-open -d \
   build/obj_so/kernel_compute.o.0.out > disasm.txt
python3 .agents/work/agent-regspill/tmp/frame.py
```

---

## 10. 实验记录（分支 `agent/regspill/spill-eval`）

> 目的：验证第 8 节各方向能否**降低 spill 并保持 `mybench -s 1` 不回退 >5%**。
> 方法：静态栈帧统计（无 GPU）+ `mybench -R 32 -s 1 -tprs 8,16,32,64,128,256 -nodes 3 -replays 2`
> 交错重复 3 次取均值；权威读数 `MT_DUMP_RES=1`。基线：`base` = 提交 `20820cc`。

### 10.1 结果总表

| 实验 | 机制 | 静态 spill（全函数栈帧和） | 内核 local（权威） | perf 变化（TPR8/16/32/64/128/256） | 判定 |
|---|---|---:|---:|---|---|
| **base** | 现状 | 3004 B（r19_big_task 1344） | 3636 B | — | — |
| **E1 全量 noinline** | `r19_l1_big` 全改 `__noinline__` | **4364 B（↑）** | — | +5.3%/+6.0%/+1.3%/−2.3%/+0.5%/+0.1% | ❌ 小档回退 >5% |
| **E1s 选择性 noinline** | 只把 `kMT≥2` 的大档经 shim 隔离 | **2464 B（↓）** r19_big_task 800 | **3092 B（−15%）** | −1.2%/+0.2%/+0.6%/−1.4%/−0.8%/0.0% | ✅ spill↓ / 中性 |
| **E2a 降 acc（L2 kBatch 2→1）** | `-DR19_L2_BATCH=1` | **1868 B（↓↓）** | — | —/—/—/—/**+13%**/**+15%** | ❌ 大档回退 |
| **E2b 降 acc（L1 MT8→4）** | `-DR19_L1_MT=4`（acc 128→64） | 3004 B（不变） | — | ~0/~0/~0/~0/**+13%**/**+13.5%** | ❌ 大档回退且不降 spill |
| **E3 COMBINE 推送 noinline** | `combine_push_*` 改 `__noinline__` | **4796 B（↑）** | — | +0.3%/−0.8%/−1.2%/+0.4%/−0.5%/−1.2% | ⚠️ 中性（spill 反升） |
| **E5 关校验和（唯一常开非计算开销）** | `-DR24_CHKACC=0` | 2980 B（−24B） | — | +0.6%/−0.5%/+0.8%/−3.6%*/**−2.0%**/**−1.2%** | ✅ 小幅提速 ≤2%（perf-only） |
| E5' 清理诊断宏 | `MX_DBG_COUNT`/`R24_DIAG_PHASE`/`R30_DIAG_*` | 默认全为 0，`#if` 编译期消除 | — | 无收益（已 0 开销） | ○ 无可做 |
| E4 地址降宽 | int64 偏移→int32 | 需改 `MegaArgs`/`R19Sched` 接口且缓冲可能 >2GB | — | 未实施（不安全/低收益） | ○ 未验证 |
| E6 spill 对齐/合并 | 让 LDL/STL 向量化 | 溢出数组已是 16B/8B native 向量；标量无法合并 | — | 未实施（无源码杠杆） | ○ 未验证 |

> *E5 的 TPR64 均值受 base 一次 677µs 离群值抬高；TPR128/256 的 −2%/−1.2% 在 3 次
> e5 重复中高度一致（978/978/979 vs base 989/1004/1003），判定为**小幅真实收益**。

### 10.2 结论

1. **降 spill 本身不会带来加速**：E1s 把内核 local 从 3636→3092 B（−15%）、
   `r19_big_task` 栈帧 1344→800 B，perf 仍**中性（±1.5%，噪声内）**。说明被溢出的值
   处于**非关键路径**，只是"占了寄存器没进热路径"。
2. **强行降 acc 会亏在访存（L1/L2 皆然）**：E2a（L2 `kBatch=1`）把 672 B spill 全消、
   E2b（L1 `MT=4`）把 acc 128→64，两者都让 **TPR128/256 +13~15%**，且 E2b 连 spill
   都没降（r24_l1_big 本就 0 栈帧，真正的 spiller 是 `r19_big_task`）。
   与 `docs/12 §9`（MT16/kBatch=1 因 A 流量翻倍被否）同因。
3. **盲目 `__noinline__` 反而增大总 spill**：E1 把 `r19_l1_big<1,*>` 也隔离，
   使 `r19_small_task` 栈帧 0→224 B，小档 +5~6%；E3 让推送调用点跨调用保活，
   使 `r24_l2_big<8,8,1>` 0→1088 B。→ 隔离只在**大档**且**调用点可保活少**时成立。
4. **唯一"提速"的是 E5（关校验和）**：校验和是 GEMM 体内唯一的常开非计算开销
   （诊断宏已全部编译期关闭，见 §10.1 E5'）。关掉后 TPR128 −2.0%、TPR256 −1.2%、
   其余噪声内，**<5% 达标**。但 `check` 依赖校验和 → 只能做 perf/check 双路径
   （模板参或 host 开关），属"可选"而非"免费"。
5. **安全可留的改动 = E1s**：把大档 L1 经 `r19_l1_big_iso`（`__noinline__`）隔离，
   小档保持内联；spill 降 ~18%，perf 中性。实现见 `src/compute_big.cuh` 宏
   `R19_L1BIG_ISO`（默认 `__noinline__`，`-DR19_L1BIG_ISO=__forceinline__` 回退基线）。
6. **E4/E6 未实施**：E4（int64→int32）要动 `MegaArgs`/`R19Sched` 接口（硬边界，需串行）
   且缓冲可能 >2GB，正确性风险高；E6 的溢出对象已是 16B/8B native 向量，标量合并
   无源码杠杆。按 §10.2.1 的结论，二者即便做了也换不来 >5%。
7. **总结论**：本轮所有"降 spill / 降开销"尝试中，**只有 E1s（降 spill，中性）与
   E5（关校验和，−2%）有效且都在 ≤5% 约束内**；其余（E1/E2a/E2b/E3）无效或回退。
   要真正提速须攻瓶颈（权重/A 流量、COMBINE 排空），而非 spill。

### 10.3 复现

```bash
# 基线（shim 内联）
KERNEL_CFLAGS="-DR19_L1BIG_ISO=__forceinline__" scripts/build.sh so mybench
# E1s（默认：大档隔离）
scripts/build.sh so mybench
# E2a / E2b / E5 变体
KERNEL_CFLAGS="-DR19_L2_BATCH=1"                scripts/build.sh so mybench
KERNEL_CFLAGS="-DR19_L1_MT=4"                   scripts/build.sh so mybench
KERNEL_CFLAGS="-DR24_CHKACC=0"                  scripts/build.sh so mybench
# 权威读数
MT_DUMP_RES=1 scripts/run.sh mybench -R 32 -s 1 -tprs 16 -nodes 1 -replays 1
# 交错 A/B（需 GPU 锁）
BENCH=<bin>/mybench scripts/run.sh mybench -R 32 -s 1 -tprs 8,16,32,64,128,256 -nodes 3 -replays 2
```

---

## 11. 已实施（261011_2018，`main` 工作区）：大档 L1 + 中档 L2 选择性 noinline 隔离

按 §8 方向 1 / §10 E1s 落地并**扩展**到 L2：

- `src/common.cuh`：新增 `R19_L1BIG_ISO`（默认 `__noinline__`）。
- `src/compute_big.cuh`：新增 `r19_l1_big_iso` shim；`r19_big_task` 内 **kMT>=2** 的 5 处
  L1 档改走 shim（小档 kMT=1 保持内联）。
- `src/compute.cuh`：新增 `l2_gemm_body_iso`（`R19_L2BODY_ISO`，默认 `__noinline__`）；
  `r19_big_task` 内 6 处中/小档 L2 body（kMTiles=2/3，acc 最多 96）改走 shim。

**结果**（本机 XC61610，`MT_DUMP_RES=1`）：内核 `local` **3096→2568 B（−17%）**，regs 仍 256；
`r19_big_task` 静态栈帧 **1344→528 B**。**perf 中性偏升**：`-s1` TPR32 **−6.8%**、其余 |Δ|≤1.4%；
`-s2 top50` mean **792.9→771.9µs（−2.6%）**。`check --smoke` PASS。
剩余溢出在 `r24_l2_big`（noinline 自包含，acc=128 结构性），再降需牺牲访存（§10 已证回退）。

