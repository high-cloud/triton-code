import torch
import torch.npu
import triton
import triton.language as tl


@triton.jit
def select_kernel(
    condition_ptr,  # 条件张量的指针 (B, H, W)
    x_ptr,  # 输入张量 x 的指针 (B, H, W)
    y_ptr,  # 输入张量 y 的指针 (B, H, W)
    output_ptr,  # 输出张量的指针 (B, H, W)
    BLOCK_B: tl.constexpr,  # batch 维度的 block 大小
    BLOCK_H: tl.constexpr,  # height 维度大小
    BLOCK_W: tl.constexpr,  # width 维度的 block 大小
):
    # 获取当前处理的 batch 索引
    pid_b = tl.program_id(0)  # batch 维度的程序 ID

    # 计算当前 block 的起始位置
    b_idx = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    h_idx = tl.arange(0, BLOCK_H)
    w_idx = tl.arange(0, BLOCK_W)

    # 计算全局索引：idx = b * (H * W) + h * W + w
    b_expanded = b_idx[:, None, None]  # [BLOCK_B, 1, 1]
    h_expanded = h_idx[None, :, None]  # [1, BLOCK_H, 1]
    w_expanded = w_idx[None, None, :]  # [1, 1, BLOCK_W]

    global_idx = b_expanded * (BLOCK_H * BLOCK_W) + h_expanded * BLOCK_W + w_expanded

    # 加载条件、x 和 y（无 mask）
    cond = tl.load(condition_ptr + global_idx)
    x_values = tl.load(x_ptr + global_idx)
    y_values = tl.load(y_ptr + global_idx)

    # 使用 select 操作：如果条件为真选择 x，否则选择 y
    # 在 triton 中，select 通过 where 实现
    result = tl.where(cond, x_values, y_values)

    # 存储结果
    tl.store(output_ptr + global_idx, result)


def triton_func(condition: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert len(condition.shape) == 3, "condition 必须是三维张量"
    assert len(x.shape) == 3, "x 必须是三维张量"
    assert len(y.shape) == 3, "y 必须是三维张量"
    assert condition.shape == x.shape == y.shape, "所有输入张量形状必须相同"
    
    B, H, W = condition.shape
    
    # 固定维度大小，确保能被 block 大小整除
    assert B == 4, "B 维度必须为 4"
    assert H == 8, "H 维度必须为 8"
    assert W == 32, "W 维度必须为 32"
    
    output = torch.empty_like(x)
    
    # 设置 block 大小（必须能整除对应维度）
    BLOCK_B = 4  # 必须能整除 B=4
    BLOCK_H = 8  # 必须能整除 H=8
    BLOCK_W = 32  # 必须能整除 W=32
    
    grid = lambda meta: (triton.cdiv(B, meta['BLOCK_B']),)
    
    select_kernel[grid](
        condition,
        x,
        y,
        output,
        BLOCK_B=BLOCK_B,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
    )
    
    return output


def torch_func(condition: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.where(condition, x, y)


if __name__ == "__main__":
    torch.manual_seed(0)
    size = (4, 8, 32)  # (B, H, W) 三维张量，固定维度大小
    condition = torch.randn(size, device="npu", dtype=torch.float32) > 0
    x = torch.randn(size, device="npu", dtype=torch.float32)
    y = torch.randn(size, device="npu", dtype=torch.float32)
    output_triton = triton_func(condition, x, y)
    output_torch = torch_func(condition, x, y)
    # 验证结果
    print(f"条件形状: {condition.shape}")
    print(f"输入形状: x={x.shape}, y={y.shape}")
    print(f"输出形状: {output_triton.shape}")
    print(f"最大误差: {torch.max(torch.abs(output_triton - output_torch)).item():.6f}")
    print(f"是否相等: {torch.allclose(output_triton, output_torch, rtol=1e-5, atol=1e-5)}")
    print(f"\n条件示例 (前2个batch, 前2个H, 前4个W):\n{condition[:2, :2, :4].int()}")
    print(f"\n输入 x 示例 (前2个batch, 前2个H, 前4个W):\n{x[:2, :2, :4]}")
    print(f"\n输入 y 示例 (前2个batch, 前2个H, 前4个W):\n{y[:2, :2, :4]}")
    print(f"\nTriton 输出示例 (前2个batch, 前2个H, 前4个W):\n{output_triton[:2, :2, :4]}")
    print(f"\nTorch 输出示例 (前2个batch, 前2个H, 前4个W):\n{output_torch[:2, :2, :4]}")

