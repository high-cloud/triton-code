import torch
import torch.npu
import triton
import triton.language as tl


@triton.jit
def triton_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
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

    input_idx = b_expanded * (BLOCK_H * BLOCK_W) + h_expanded * BLOCK_W + w_expanded

    x_values = tl.load(x_ptr + input_idx)
    y_values = tl.load(y_ptr + input_idx)

    interleaved = tl.interleave(x_values, y_values)

    w_out_idx = tl.arange(0, BLOCK_W * 2)
    w_out_expanded = w_out_idx[None, None, :]
    
    output_idx = b_expanded * (BLOCK_H * BLOCK_W * 2) + h_expanded * (BLOCK_W * 2) + w_out_expanded

    tl.store(output_ptr + output_idx, interleaved)


def triton_func(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert len(x.shape) == 3, "x 必须是三维张量"
    assert len(y.shape) == 3, "y 必须是三维张量"
    assert x.shape == y.shape, "x 和 y 的形状必须相同"
    B, H, W = x.shape
    
    output = torch.empty((B, H, 2 * W), device=x.device, dtype=x.dtype)
    
    BLOCK_B = 4
    BLOCK_H = 8
    BLOCK_W = 17
    
    grid = lambda meta: (triton.cdiv(B, meta['BLOCK_B']),)
    
    triton_kernel[grid](
        x,
        y,
        output,
        BLOCK_B=BLOCK_B,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
    )
    
    return output


def torch_func(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    stacked = torch.stack([x, y], dim=0)
    interleaved = stacked.permute(1, 2, 3, 0).reshape(x.shape[0], x.shape[1], -1)
    return interleaved


if __name__ == "__main__":
    torch.manual_seed(0)
    size = (8, 8, 17)
    x = torch.randn(size, device="npu", dtype=torch.float32)
    y = torch.randn(size, device="npu", dtype=torch.float32)
    output_triton = triton_func(x, y)
    output_torch = torch_func(x, y)
    print(f"输入形状: x={x.shape}, y={y.shape}")
    print(f"输出形状: {output_triton.shape}")
    print(f"最大误差: {torch.max(torch.abs(output_triton - output_torch)).item():.6f}")
    print(f"是否相等: {torch.allclose(output_triton, output_torch, rtol=1e-5, atol=1e-5)}")
    print(f"\n输入 x 示例 (前2个batch, 前2个H, 前4个W):\n{x[:2, :2, :4]}")
    print(f"\n输入 y 示例 (前2个batch, 前2个H, 前4个W):\n{y[:2, :2, :4]}")
    print(f"\nTriton 输出示例 (前2个batch, 前2个H, 前8个W):\n{output_triton[:2, :2, :8]}")
    print(f"\nTorch 输出示例 (前2个batch, 前2个H, 前8个W):\n{output_torch[:2, :2, :8]}")

