import torch
import torch.npu
import triton
import triton.language as tl


@triton.jit
def triton_kernel(
    input_ptr,
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

    output_idx = b_expanded * (H * W) + h_expanded * W + w_expanded

    values = tl.load(input_ptr + output_idx)
    values_flipped = tl.flip(values, dim=2)

    tl.store(output_ptr + output_idx, values_flipped)


def triton_func(x: torch.Tensor) -> torch.Tensor:
    assert len(x.shape) == 3, "输入必须是三维张量"
    B, H, W = x.shape
    output = torch.empty_like(x)

    BLOCK_B = 4
    BLOCK_H = 8
    BLOCK_W = 17

    assert H == BLOCK_H
    assert W == BLOCK_W

    grid = lambda meta: (triton.cdiv(B, meta['BLOCK_B']),)

    triton_kernel[grid](
        x,
        output,
        BLOCK_B=BLOCK_B,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
    )

    return output


def torch_func(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(x, dims=[2])


if __name__ == "__main__":
    torch.manual_seed(0)
    size = (12, 8, 17)
    x = torch.randn(size, device="npu", dtype=torch.float32)
    output_triton = triton_func(x)
    output_torch = torch_func(x)
    print(f"输入形状: x={x.shape}")
    print(f"输出形状: {output_triton.shape}")
    print(f"最大误差: {torch.max(torch.abs(output_triton - output_torch)).item():.6f}")
    print(f"是否相等: {torch.allclose(output_triton, output_torch, rtol=1e-5, atol=1e-5)}")
