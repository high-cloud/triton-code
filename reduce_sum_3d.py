import torch
import torch.npu
import triton
import triton.language as tl


@triton.jit
def reduce_sum_kernel(
    input_ptr,  # 输入张量的指针
    output_ptr,  # 输出张量的指针
    H: tl.constexpr,  # height 维度大小
    W: tl.constexpr,  # width 维度大小（最后一根轴）
    BLOCK_B: tl.constexpr,  # batch 维度的 block 大小
    BLOCK_H: tl.constexpr,  # height 维度的 block 大小
    BLOCK_W: tl.constexpr,  # width 维度的 block 大小（reduce 维度）
):
    # 获取当前处理的 batch 和 height 索引
    pid_b = tl.program_id(0)  # batch 维度的程序 ID
    pid_h = tl.program_id(1)  # height 维度的程序 ID
    
    # 计算当前 block 的起始位置
    b_idx = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    h_idx = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    w_idx = tl.arange(0, BLOCK_W)
    
    # 计算全局索引：idx = b * (H * W) + h * W + w
    b_expanded = b_idx[:, None, None]  # [BLOCK_B, 1, 1]
    h_expanded = h_idx[None, :, None]  # [1, BLOCK_H, 1]
    w_expanded = w_idx[None, None, :]  # [1, 1, BLOCK_W]
    
    global_idx = b_expanded * (H * W) + h_expanded * W + w_expanded
    
    # 加载数据（无 mask）
    values = tl.load(input_ptr + global_idx)
    
    # 对 W 维度求和
    acc = tl.sum(values, axis=2)
    
    # 存储结果：output[b, h] = sum(input[b, h, :])
    output_idx = b_idx[:, None] * H + h_idx[None, :]
    tl.store(output_ptr + output_idx, acc)


def triton_func(x: torch.Tensor) -> torch.Tensor:
    assert len(x.shape) == 3, "输入必须是三维张量"
    B, H, W = x.shape
    
    # 固定维度大小，确保能被 block 大小整除
    assert B == 4, "B 维度必须为 4"
    assert H == 8, "H 维度必须为 8"
    assert W == 32, "W 维度必须为 32"
    
    output = torch.empty((B, H), device=x.device, dtype=x.dtype)
    
    # 设置 block 大小（必须能整除对应维度）
    BLOCK_B = 4  # 必须能整除 B=4
    BLOCK_H = 8  # 必须能整除 H=8
    BLOCK_W = 32  # 必须能整除 W=32
    
    grid = lambda meta: (
        triton.cdiv(B, meta['BLOCK_B']),
        triton.cdiv(H, meta['BLOCK_H']),
    )
    
    reduce_sum_kernel[grid](
        x,
        output,
        H=H,
        W=W,
        BLOCK_B=BLOCK_B,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
    )
    
    return output


def torch_func(x: torch.Tensor) -> torch.Tensor:
    return torch.sum(x, dim=-1)


if __name__ == "__main__":
    torch.manual_seed(0)
    size = (4, 8, 32)  # (B, H, W) 三维张量，固定维度大小
    x = torch.randn(size, device="npu", dtype=torch.float32)
    output_triton = triton_func(x)
    output_torch = torch_func(x)
    # 验证结果
    print(f"输入形状: x={x.shape}")
    print(f"输出形状: {output_triton.shape}")
    print(f"最大误差: {torch.max(torch.abs(output_triton - output_torch)).item():.6f}")
    print(f"是否相等: {torch.allclose(output_triton, output_torch, rtol=1e-5, atol=1e-5)}")
    print(f"\nTriton 输出示例:\n{output_triton[:2, :4]}")
    print(f"\nTorch 输出示例:\n{output_torch[:2, :4]}")

