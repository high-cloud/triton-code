import torch
import torch.npu
import triton
import triton.language as tl


@triton.jit
def interleave_kernel(
    x_ptr,  # 输入张量 x 的指针 (B, H, W)
    y_ptr,  # 输入张量 y 的指针 (B, H, W)
    output_ptr,  # 输出张量的指针 (B, H, 2*W)
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

    # x 和 y 的索引：x[b, h, w], y[b, h, w]
    input_idx = b_expanded * (BLOCK_H * BLOCK_W) + h_expanded * BLOCK_W + w_expanded

    # 加载数据（无 mask）
    x_values = tl.load(x_ptr + input_idx)
    y_values = tl.load(y_ptr + input_idx)

    # interleave: output[b, h, 2*w] = x[b, h, w], output[b, h, 2*w+1] = y[b, h, w]
    # 输出索引：output[b, h, 2*w] 和 output[b, h, 2*w+1]
    output_w_even = w_expanded * 2  # [1, 1, BLOCK_W] -> 偶数位置
    output_w_odd = w_expanded * 2 + 1  # [1, 1, BLOCK_W] -> 奇数位置
    
    output_idx_even = b_expanded * (BLOCK_H * BLOCK_W * 2) + h_expanded * (BLOCK_W * 2) + output_w_even
    output_idx_odd = b_expanded * (BLOCK_H * BLOCK_W * 2) + h_expanded * (BLOCK_W * 2) + output_w_odd

    # 存储交错结果
    tl.store(output_ptr + output_idx_even, x_values)
    tl.store(output_ptr + output_idx_odd, y_values)


def triton_func(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert len(x.shape) == 3, "x 必须是三维张量"
    assert len(y.shape) == 3, "y 必须是三维张量"
    assert x.shape == y.shape, "x 和 y 的形状必须相同"
    B, H, W = x.shape
    
    # 固定维度大小，确保能被 block 大小整除
    assert B == 4, "B 维度必须为 4"
    assert H == 8, "H 维度必须为 8"
    assert W == 32, "W 维度必须为 32"
    
    # 输出形状：(B, H, 2*W)
    output = torch.empty((B, H, 2 * W), device=x.device, dtype=x.dtype)
    
    # 设置 block 大小（必须能整除对应维度）
    BLOCK_B = 4  # 必须能整除 B=4
    BLOCK_H = 8  # 必须能整除 H=8
    BLOCK_W = 32  # 必须能整除 W=32
    
    grid = lambda meta: (triton.cdiv(B, meta['BLOCK_B']),)
    
    interleave_kernel[grid](
        x,
        y,
        output,
        BLOCK_B=BLOCK_B,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
    )
    
    return output


def torch_func(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # 使用 stack 和 reshape 实现 interleave
    # stack 得到 (2, B, H, W)，然后 transpose 和 reshape
    stacked = torch.stack([x, y], dim=0)  # (2, B, H, W)
    # 交换维度得到 (B, H, W, 2)，然后 reshape 成 (B, H, 2*W)
    interleaved = stacked.permute(1, 2, 3, 0).reshape(x.shape[0], x.shape[1], -1)
    return interleaved


if __name__ == "__main__":
    torch.manual_seed(0)
    size = (4, 8, 32)  # (B, H, W) 三维张量
    x = torch.randn(size, device="npu", dtype=torch.float32)
    y = torch.randn(size, device="npu", dtype=torch.float32)
    output_triton = triton_func(x, y)
    output_torch = torch_func(x, y)
    # 验证结果
    print(f"输入形状: x={x.shape}, y={y.shape}")
    print(f"输出形状: {output_triton.shape}")
    print(f"最大误差: {torch.max(torch.abs(output_triton - output_torch)).item():.6f}")
    print(f"是否相等: {torch.allclose(output_triton, output_torch, rtol=1e-5, atol=1e-5)}")
    print(f"\n输入 x 示例 (前2个batch, 前2个H, 前4个W):\n{x[:2, :2, :4]}")
    print(f"\n输入 y 示例 (前2个batch, 前2个H, 前4个W):\n{y[:2, :2, :4]}")
    print(f"\nTriton 输出示例 (前2个batch, 前2个H, 前8个W):\n{output_triton[:2, :2, :8]}")
    print(f"\nTorch 输出示例 (前2个batch, 前2个H, 前8个W):\n{output_torch[:2, :2, :8]}")

