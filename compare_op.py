import torch
import torch.npu
import triton
import triton.language as tl


@triton.jit
def compare_kernel(
    input_ptr,  # 输入张量 x 的指针
    output_ptr,  # 输出张量的指针
    threshold: tl.constexpr,  # 比较阈值
    BLOCK_Y: tl.constexpr,
    BLOCK_X: tl.constexpr,
):
    pid = tl.program_id(0)  # 程序 ID
    
    blk_beg = pid * BLOCK_Y
    x1 = tl.arange(0, BLOCK_X)
    y0 = blk_beg + tl.arange(0, BLOCK_Y)
    idx = x1[None, :] + y0[:, None] * BLOCK_X
    tmp0 = tl.load(input_ptr + idx)
    # 使用 compare 操作：大于阈值的元素为 True，否则为 False
    tmp1 = tmp0 > threshold
    # 将布尔值转换为 float32
    tmp2 = tmp1.to(tl.float32)
    tl.store(output_ptr + idx, tmp2)


def triton_func(x0: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    output = torch.empty_like(x0, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(x0.shape[0], meta['BLOCK_Y']),)
    compare_kernel[grid](
        x0,
        output,
        threshold=threshold,
        BLOCK_Y=16,
        BLOCK_X=32,
    )
    return output


def torch_func(x0: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    return (x0 > threshold).to(torch.float32)


if __name__ == "__main__":
    torch.manual_seed(0)
    size = (16, 32)
    threshold = 0.5
    x = torch.randn(size, device="npu", dtype=torch.float32)
    output_triton = triton_func(x, threshold)
    output_torch = torch_func(x, threshold)
    # 验证结果
    print(f"输入形状: x={x.shape}")
    print(f"阈值: {threshold}")
    print(f"输出形状: {output_triton.shape}")
    print(f"最大误差: {torch.max(torch.abs(output_triton - output_torch)).item():.6f}")
    print(f"是否相等: {torch.allclose(output_triton, output_torch)}")

