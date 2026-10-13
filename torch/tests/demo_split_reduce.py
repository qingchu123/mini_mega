"""demo_split_reduce.py — 演示用拆分 op 隐藏 REDUCE 的跨 rank 等待。

背景：`fp8_mega_moe` = compute（前序+任务循环+COMBINE）+ REDUCE（topk 加权归约）。
REDUCE 必须等**所有 rank** 的 COMBINE 完成标记，快 rank 会先到而空等慢 rank。
新增的 `fp8_mega_moe_compute` / `fp8_mega_moe_reduce` 把两段拆成独立 op，框架层
可在两者之间插入**与本 MoE 输出无关**的 kernel，把这段空等重叠隐藏。

本 demo 对比：
  fused       : fp8_mega_moe(...)                      融合
  split       : compute → reduce                        拆分（无插入）
  split+work  : compute → independent_work() → reduce   拆分 + 插入无关工作
若 split+work 与 split 的耗时差 ≈ 0（而 independent_work 单独计时很大），说明
独立工作的时间被 REDUCE 的跨 rank 等待隐藏了。

运行（32 卡，例如 202 的 cl_test 容器内）：
  torchrun --nproc_per_node=32 demo_split_reduce.py --iters 50 --work-mb 64
"""
import argparse
import time

import torch
import torch.distributed as dist

import mxmegakernel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=6144)
    ap.add_argument("--intermediate", type=int, default=2048)
    ap.add_argument("--experts", type=int, default=256)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--work-mb", type=int, default=64)   # 独立工作规模（MB）
    ap.add_argument("--backend", type=str, default="nccl")
    args = ap.parse_args()

    dist.init_process_group(args.backend)
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    group = dist.group.WORLD

    epr = args.experts // world
    buf = mxmegakernel.get_symm_buffer_for_mega_moe(
        group, args.experts, args.tokens, args.topk,
        args.hidden, args.intermediate, expert_window=None)

    # 均衡路由 + fp8 权重（zeros 即可，本 demo 只看时延）
    idx = (torch.arange(args.tokens, device=dev).unsqueeze(1)
           + torch.arange(args.topk, device=dev).unsqueeze(0))
    idx = (idx % args.experts).to(buf.topk_idx.dtype)
    buf.topk_idx[:args.tokens].copy_(idx)
    buf.topk_weights[:args.tokens].fill_(1.0 / args.topk)
    l1 = torch.zeros((epr, 2 * args.intermediate, args.hidden),
                     dtype=torch.float8_e4m3fn, device=dev)
    l2 = torch.zeros((epr, args.hidden, args.intermediate),
                     dtype=torch.float8_e4m3fn, device=dev)
    tl1, tl2 = mxmegakernel.transform_weights_for_mega_moe(l1, l2)
    y = torch.empty((args.tokens, args.hidden), dtype=torch.bfloat16, device=dev)

    # 与本 MoE 完全无关的"独立工作"（示例：链式 GEMM）。
    W = torch.randn((1024, 1024), device=dev) * 0.01
    rows = max(1024, args.work_mb * 1024 * 1024 // (1024 * 4))
    Xw = torch.randn((rows, 1024), device=dev)

    def independent_work():
        for _ in range(4):
            Xw.copy_(Xw @ W)

    # 预热（触发跨 rank IPC 懒映射）
    for _ in range(3):
        mxmegakernel.fp8_mega_moe(y, tl1, tl2, buf, activation_clamp=1e30)
    torch.cuda.synchronize()
    dist.barrier()

    def timed(fn):
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / args.iters * 1e3   # ms

    fused = timed(lambda: mxmegakernel.fp8_mega_moe(
        y, tl1, tl2, buf, activation_clamp=1e30))
    split = timed(lambda: (
        mxmegakernel.fp8_mega_moe_compute(
            tl1, tl2, buf, args.tokens, activation_clamp=1e30),
        mxmegakernel.fp8_mega_moe_reduce(y, buf, args.tokens)))
    split_work = timed(lambda: (
        mxmegakernel.fp8_mega_moe_compute(
            tl1, tl2, buf, args.tokens, activation_clamp=1e30),
        independent_work(),
        mxmegakernel.fp8_mega_moe_reduce(y, buf, args.tokens)))
    work_only = timed(independent_work)

    if rank == 0:
        print(f"[demo] fused={fused:.4f}ms  split={split:.4f}ms  "
              f"split+work={split_work:.4f}ms  work_only={work_only:.4f}ms")
        print(f"[demo] 插入工作净增 = {split_work - split:+.4f}ms "
              f"(work_only={work_only:.4f}ms；净增远小于 work_only → 被 REDUCE 等待隐藏)")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
