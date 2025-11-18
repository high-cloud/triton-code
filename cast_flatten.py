import torch
import torch.npu
import triton
import triton.language as tl


@triton.jit
def cast_kernel(
    input_ptr,  # 输入张量 x 的指针
    output_ptr,  # 输出张量的指针
    BLOCK_Y: tl.constexpr,
    BLOCK_X: tl.constexpr,
):
    pid = tl.program_id(0)  # 程序 ID
    
    blk_beg = pid * BLOCK_Y
    X1 = tl.arange(0, BLOCK_X)
    y0 = blk_beg + tl.arange(0, BLOCK_Y)
    idx = x1[None, :] + y0[:, None] * BLOCK_X
    tmp0 = tl.load(input_ptr + idx)
    tmp1 = tmp0.to(tl.float16)
    tl.store(output_ptr + idx, tmp1)


def triton_func(x0: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x0)
    grid = lambda meta: (triton.cdiv(x0.shape[0], meta['BLOCK_Y']),)
    cast_kernel[grid](
        x0,
        output,
        BLOCK_Y=16,
        BLOCK_X=32,
    )
    return output


def torch_func(x0: torch.Tensor) -> torch.Tensor:
    return x0.to(torch.float16)


if __name__ == "__main__":
    torch.manual_seed(0)
    size = (16, 32)
    x0 = torch.randn(size, device="npu", dtype=torch.float32)
    output_triton = triton_func(x0)
    output_torch = torch_func(x0)
    # 验证结果
    print(f"输入形状: x={x.shape}, y={y.shape}")
    print(f"输出形状: {output_triton.shape}")
    print(f"最大误差: {torch.max(torch.abs(output_triton - output_torch)).item():.6f}")
    print(f"是否相等: {torch.allclose(output_triton, output_torch)}")

