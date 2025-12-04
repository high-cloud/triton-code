import torch
import torch.npu
import triton
import triton.language as tl


@triton.jit
def triton_kernel(
    input_ptr,
    output_ptr,
    threshold,
    max_iterations,
    BLOCK_B: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_b = tl.program_id(0)

    b_idx = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    h_idx = tl.arange(0, BLOCK_H)
    w_idx = tl.arange(0, BLOCK_W)

    b_expanded = b_idx[:, None, None]
    h_expanded = h_idx[None, :, None]
    w_expanded = w_idx[None, None, :]

    global_idx = b_expanded * (BLOCK_H * BLOCK_W) + h_expanded * BLOCK_W + w_expanded

    values = tl.load(input_ptr + global_idx)

    result = values
    iteration = 0
    target = 1.0

    while iteration < max_iterations:
        prev_result = result
        result = result * 0.9 + 0.1
        diff = tl.abs(result - prev_result)
        iteration = iteration + 1
        if tl.max(diff) < threshold:
            break

    tl.store(output_ptr + global_idx, result)


def triton_func(x: torch.Tensor, threshold: float = 0.001, max_iterations: int = 100) -> torch.Tensor:
    assert len(x.shape) == 3, "输入必须是三维张量"
    B, H, W = x.shape
    
    output = torch.empty_like(x)
    
    BLOCK_B = 4
    BLOCK_H = 8
    BLOCK_W = 32
    
    assert H == BLOCK_H
    assert W == BLOCK_W
    
    grid = lambda meta: (triton.cdiv(B, meta['BLOCK_B']),)
    
    triton_kernel[grid](
        x,
        output,
        threshold=threshold,
        max_iterations=max_iterations,
        BLOCK_B=BLOCK_B,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
    )
    
    return output


def torch_func(x: torch.Tensor, threshold: float = 0.001, max_iterations: int = 100) -> torch.Tensor:
    result = x.clone()
    iteration = 0
    target = 1.0
    
    while iteration < max_iterations:
        prev_result = result.clone()
        result = result * 0.9 + 0.1
        diff = torch.abs(result - prev_result)
        iteration = iteration + 1
        if torch.max(diff) < threshold:
            break
    
    return result


if __name__ == "__main__":
    torch.manual_seed(0)
    size = (12, 8, 32)
    
    print("=" * 60)
    print("测试: while 循环直到收敛")
    print("=" * 60)
    x = torch.randn(size, device="npu", dtype=torch.float32)
    threshold = 0.001
    max_iter = 100
    output_triton = triton_func(x, threshold, max_iter)
    output_torch = torch_func(x, threshold, max_iter)
    print(f"输入形状: x={x.shape}")
    print(f"收敛阈值: {threshold}")
    print(f"最大迭代次数: {max_iter}")
    print(f"输出形状: {output_triton.shape}")
    print(f"最大误差: {torch.max(torch.abs(output_triton - output_torch)).item():.6f}")
    print(f"是否相等: {torch.allclose(output_triton, output_torch, rtol=1e-4, atol=1e-4)}")
    print(f"\n输入示例 (前2个batch, 前2个H, 前4个W):\n{x[:2, :2, :4]}")
    print(f"\nTriton 输出示例 (前2个batch, 前2个H, 前4个W):\n{output_triton[:2, :2, :4]}")
    print(f"\nTorch 输出示例 (前2个batch, 前2个H, 前4个W):\n{output_torch[:2, :2, :4]}")

