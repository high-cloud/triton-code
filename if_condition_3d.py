import torch
import torch.npu
import triton
import triton.language as tl


@triton.jit
def triton_kernel(
    input_ptr,  # 输入张量的指针 (B, H, W)
    output_ptr,  # 输出张量的指针 (B, H, W)
    threshold,  # 阈值，用于条件判断
    BLOCK_B: tl.constexpr,  # batch 维度的 block 大小
    BLOCK_H: tl.constexpr,  # height 维度大小
    BLOCK_W: tl.constexpr,  # width 维度的 block 大小
):
    # 固定维度大小
    B = 4
    H = 8
    W = 32

    # 获取当前处理的 batch 索引
    pid_b = tl.program_id(0)  # batch 维度的程序 ID

    # 计算当前 block 的起始位置
    b_idx = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    h_idx = tl.arange(0, BLOCK_H)
    w_idx = tl.arange(0, BLOCK_W)

    # 计算全局索引
    b_expanded = b_idx[:, None, None]  # [BLOCK_B, 1, 1]
    h_expanded = h_idx[None, :, None]  # [1, BLOCK_H, 1]
    w_expanded = w_idx[None, None, :]  # [1, 1, BLOCK_W]

    global_idx = b_expanded * (BLOCK_H * BLOCK_W) + h_expanded * BLOCK_W + w_expanded

    # 加载数据（无 mask）
    values = tl.load(input_ptr + global_idx)

    # 使用 scf.if 的条件分支：根据阈值选择不同的操作
    # 如果 threshold > 0，则对值进行平方；否则保持原值
    if threshold > 0.0:
        # 条件为真时的操作：平方
        result = values * values
    else:
        # 条件为假时的操作：保持原值
        result = values

    # 存储结果
    tl.store(output_ptr + global_idx, result)


def triton_func(x: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    assert len(x.shape) == 3, "输入必须是三维张量"
    B, H, W = x.shape
    
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
    
    triton_kernel[grid](
        x,
        output,
        threshold=threshold,
        BLOCK_B=BLOCK_B,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
    )
    
    return output


def torch_func(x: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    # 使用条件分支：如果 threshold > 0，则平方；否则保持原值
    if threshold > 0.0:
        return x * x
    else:
        return x


if __name__ == "__main__":
    torch.manual_seed(0)
    size = (4, 8, 32)  # (B, H, W) 三维张量，固定维度大小
    
    # 测试 threshold > 0 的情况
    print("=" * 60)
    print("测试 1: threshold > 0 (应该对值进行平方)")
    print("=" * 60)
    x1 = torch.randn(size, device="npu", dtype=torch.float32)
    threshold1 = 0.5
    output_triton1 = triton_func(x1, threshold1)
    output_torch1 = torch_func(x1, threshold1)
    print(f"输入形状: x={x1.shape}")
    print(f"阈值: {threshold1}")
    print(f"输出形状: {output_triton1.shape}")
    print(f"最大误差: {torch.max(torch.abs(output_triton1 - output_torch1)).item():.6f}")
    print(f"是否相等: {torch.allclose(output_triton1, output_torch1, rtol=1e-5, atol=1e-5)}")
    print(f"\n输入示例 (前2个batch, 前2个H, 前4个W):\n{x1[:2, :2, :4]}")
    print(f"\nTriton 输出示例 (前2个batch, 前2个H, 前4个W):\n{output_triton1[:2, :2, :4]}")
    print(f"\nTorch 输出示例 (前2个batch, 前2个H, 前4个W):\n{output_torch1[:2, :2, :4]}")
    
    # 测试 threshold <= 0 的情况
    print("\n" + "=" * 60)
    print("测试 2: threshold <= 0 (应该保持原值)")
    print("=" * 60)
    x2 = torch.randn(size, device="npu", dtype=torch.float32)
    threshold2 = -0.5
    output_triton2 = triton_func(x2, threshold2)
    output_torch2 = torch_func(x2, threshold2)
    print(f"输入形状: x={x2.shape}")
    print(f"阈值: {threshold2}")
    print(f"输出形状: {output_triton2.shape}")
    print(f"最大误差: {torch.max(torch.abs(output_triton2 - output_torch2)).item():.6f}")
    print(f"是否相等: {torch.allclose(output_triton2, output_torch2, rtol=1e-5, atol=1e-5)}")
    print(f"\n输入示例 (前2个batch, 前2个H, 前4个W):\n{x2[:2, :2, :4]}")
    print(f"\nTriton 输出示例 (前2个batch, 前2个H, 前4个W):\n{output_triton2[:2, :2, :4]}")
    print(f"\nTorch 输出示例 (前2个batch, 前2个H, 前4个W):\n{output_torch2[:2, :2, :4]}")

