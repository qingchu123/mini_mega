#pragma once

// 单一来源：每卡 kernel grid（= block 数）。C600-U 当前为 28 SM，缺省 28；
// 换卡时用 -DKGRID_OVERRIDE=<SM 数> 或 MXK_KERNEL_GRID=<SM 数> 重新构建
// （so、wheel、torch 桥接必须同一个值，test_kernel_grid.py 会校验）。
#ifndef KGRID_OVERRIDE
#define KGRID_OVERRIDE 28
#endif
constexpr int kGridBlocks = KGRID_OVERRIDE;
