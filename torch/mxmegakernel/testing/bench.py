import os
import sys
import torch
from typing import Callable, Optional


def bench(fn, num_warmups: int = 5, num_tests: int = 10,
          high_precision: bool = False):
    # Flush L2 cache with 256 MB data
    torch.cuda.synchronize()
    cache = torch.empty(int(256e6 // 4), dtype=torch.int, device='cuda')
    cache.zero_()

    # Warmup
    for _ in range(num_warmups):
        fn()

    # Add a large kernel to eliminate the CPU launch overhead
    if high_precision:
        x = torch.randn((8192, 8192), dtype=torch.float, device='cuda')
        y = torch.randn((8192, 8192), dtype=torch.float, device='cuda')
        x @ y

    # Testing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for i in range(num_tests):
        fn()
    end_event.record()
    torch.cuda.synchronize()

    return start_event.elapsed_time(end_event) / num_tests / 1e3


class empty_suppress:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


class suppress_stdout_stderr:
    def __enter__(self):
        self.outnull_file = open(os.devnull, 'w')
        self.errnull_file = open(os.devnull, 'w')

        self.old_stdout_fileno_undup = sys.stdout.fileno()
        self.old_stderr_fileno_undup = sys.stderr.fileno()

        self.old_stdout_fileno = os.dup(sys.stdout.fileno())
        self.old_stderr_fileno = os.dup(sys.stderr.fileno())

        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

        os.dup2(self.outnull_file.fileno(), self.old_stdout_fileno_undup)
        os.dup2(self.errnull_file.fileno(), self.old_stderr_fileno_undup)

        sys.stdout = self.outnull_file
        sys.stderr = self.errnull_file
        return self

    def __exit__(self, *_):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

        os.dup2(self.old_stdout_fileno, self.old_stdout_fileno_undup)
        os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)

        os.close(self.old_stdout_fileno)
        os.close(self.old_stderr_fileno)

        self.outnull_file.close()
        self.errnull_file.close()


# MXMOEMega 的 mx_test harness 用融合链名 "mega_moe" 查询计时；mini_mega 的
# launch_full 由主 kernel + REDUCE 两个 __global__ 组成，这里把链名映射到实际
# 发射的 kernel 名（profiler 表按 kernel 名逐行聚合）。
_KERNEL_CHAIN_ALIASES = {
    'mega_moe': ('mega_kernel', 'mega_reduce_kernel'),
}


def bench_kineto(fn, kernel_names, num_tests: int = 30,
                 suppress_kineto_output: bool = False,
                 trace_path: str = None, flush_l2: bool = True,
                 with_multiple_kernels: bool = False,
                 barrier: Optional[Callable] = None,
                 prewarm: bool = True):
    assert isinstance(kernel_names, str) or isinstance(kernel_names, tuple)
    is_tuple = isinstance(kernel_names, tuple)

    # Skip profiling
    # Conflict with Nsight Systems, Nsight Compute and Compute Sanitizer
    if int(os.environ.get('DG_USE_NVIDIA_TOOLS', 0)):
        return (1, ) * len(kernel_names) if is_tuple else 1

    # By default, flush L2 with an excessive 8 GB memset to give the GPU some (literal) chill time without full idle
    flush_l2_size = int(8e9 // 4)

    # For some auto-tuning kernels with prints. 上游 harness 对已预热/已 capture
    # 的 graph 传 prewarm=False，跳过 profiler 外这次多余调用；profiler 自己的
    # 丢弃窗口仍会吸收 CUPTI 首次挂载开销，计时窗口不受影响。
    if prewarm:
        fn()

    # Profile
    suppress = suppress_stdout_stderr if suppress_kineto_output else empty_suppress
    with suppress():
        schedule = torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1)
        profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA], schedule=schedule, acc_events=True)
        with profiler:
            for i in range(2):
                for _ in range(num_tests):
                    if flush_l2:
                        torch.empty(flush_l2_size, dtype=torch.int, device='cuda').zero_()
                    if barrier is not None:
                        # NOTES: use a large kernel and a barrier to eliminate the unbalanced CPU launch overhead
                        # noinspection PyProtectedMember
                        torch.cuda._sleep(int(2e7))  # ~10ms
                        barrier()
                    fn()
                torch.cuda.synchronize()
                profiler.step()

    # Parse the profiling table
    prof_lines = profiler.key_averages().table(sort_by='cuda_time_total', max_name_column_width=100).split('\n')
    if is_tuple:
        kernel_names = tuple(kernel_names)
    else:
        kernel_names = _KERNEL_CHAIN_ALIASES.get(kernel_names, (kernel_names,))

    # Save chrome traces
    if trace_path is not None:
        profiler.export_chrome_trace(trace_path)

    # Return average kernel times
    # 表中每行倒数第二列是该 kernel 的逐实例均值（CUDA time avg）。一个查询名
    # 可能命中多行（融合链的每个 __global__ 一行）：把各内核的逐实例均值相加，
    # 得到每次 fn() 的设备时间；只命中一行时与旧口径逐位一致。
    units = {'ms': 1e3, 'us': 1e6}

    def _line_time(line: str) -> float:
        time_str = line.split()[-2]
        for unit, scale in units.items():
            if unit in time_str:
                return float(time_str.replace(unit, '')) / scale
        return 0.0

    def _name_time(name: str) -> float:
        matched = [line for line in prof_lines if name in line]
        if not matched:
            print(f'[bench_kineto] 警告：profiler 表中没有匹配 {name!r} 的 kernel')
        return sum(_line_time(line) for line in matched)

    if is_tuple:
        return tuple(_name_time(name) for name in kernel_names)
    return sum(_name_time(name) for name in kernel_names)
