import torch
import torch.npu
import triton
import triton.language as tl


@triton.jit
def triton_kernel(
    input_ptr,  # 输入张量的指针 (B, H, W)
    output_ptr,  # 输出张量的指针 (B, H, W)
    num_iterations,  # 迭代次数
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

    # 计算全局索引
    b_expanded = b_idx[:, None, None]  # [BLOCK_B, 1, 1]
    h_expanded = h_idx[None, :, None]  # [1, BLOCK_H, 1]
    w_expanded = w_idx[None, None, :]  # [1, 1, BLOCK_W]

    global_idx = b_expanded * (BLOCK_H * BLOCK_W) + h_expanded * BLOCK_W + w_expanded

    # 加载初始数据（无 mask）
    values = tl.load(input_ptr + global_idx)

    # 使用 scf.for 循环：迭代计算
    # 每次迭代：value = value * 0.9 + 0.1
    # 这相当于对值进行 num_iterations 次平滑操作
    result = values
    for i in range(num_iterations):
        # 循环体：每次迭代更新值
        result = result * 0.9 + 0.1

    # 存储结果
    tl.store(output_ptr + global_idx, result)


def triton_func(x: torch.Tensor, num_iterations: int = 5) -> torch.Tensor:
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
        num_iterations=num_iterations,
        BLOCK_B=BLOCK_B,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
    )
    
    return output


def torch_func(x: torch.Tensor, num_iterations: int = 5) -> torch.Tensor:
    # 使用循环：每次迭代 value = value * 0.9 + 0.1
    result = x.clone()
    for i in range(num_iterations):
        result = result * 0.9 + 0.1
    return result


if __name__ == "__main__":
    torch.manual_seed(0)
    size = (4, 8, 32)  # (B, H, W) 三维张量，固定维度大小
    
    # 测试不同的迭代次数
    for num_iter in [1, 3, 5]:
        print("=" * 60)
        print(f"测试: num_iterations = {num_iter}")
        print("=" * 60)
        x = torch.randn(size, device="npu", dtype=torch.float32)
        output_triton = triton_func(x, num_iter)
        output_torch = torch_func(x, num_iter)
        print(f"输入形状: x={x.shape}")
        print(f"迭代次数: {num_iter}")
        print(f"输出形状: {output_triton.shape}")
        print(f"最大误差: {torch.max(torch.abs(output_triton - output_torch)).item():.6f}")
        print(f"是否相等: {torch.allclose(output_triton, output_torch, rtol=1e-5, atol=1e-5)}")
        print(f"\n输入示例 (前2个batch, 前2个H, 前4个W):\n{x[:2, :2, :4]}")
        print(f"\nTriton 输出示例 (前2个batch, 前2个H, 前4个W):\n{output_triton[:2, :2, :4]}")
        print(f"\nTorch 输出示例 (前2个batch, 前2个H, 前4个W):\n{output_torch[:2, :2, :4]}")
        print()

