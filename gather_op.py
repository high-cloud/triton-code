import torch
import torch.npu
import triton
import triton.language as tl


@triton.jit
def gather_kernel(
    input_ptr,  # 输入张量的指针
    index_ptr,  # 索引张量的指针
    output_ptr,  # 输出张量的指针
    input_size: tl.constexpr,  # 输入张量的大小
    BLOCK_Y: tl.constexpr,
    BLOCK_X: tl.constexpr,
):
    pid = tl.program_id(0)  # 程序 ID
    
    blk_beg = pid * BLOCK_Y
    x1 = tl.arange(0, BLOCK_X)
    y0 = blk_beg + tl.arange(0, BLOCK_Y)
    idx = x1[None, :] + y0[:, None] * BLOCK_X
    
    # 加载索引
    indices = tl.load(index_ptr + idx)
    # 确保索引在有效范围内
    indices = tl.minimum(tl.maximum(indices, 0), input_size - 1)
    
    # 使用 gather 操作：根据索引从输入张量中收集元素
    # 在 triton 中，gather 通过索引访问实现
    gathered = tl.load(input_ptr + indices)
    
    tl.store(output_ptr + idx, gathered)


def triton_func(x0: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    assert indices.dtype == torch.int64 or indices.dtype == torch.int32, "索引必须是整数类型"
    assert indices.shape == x0.shape, "索引形状必须与输入形状相同"
    output = torch.empty_like(x0)
    input_size = x0.numel()
    grid = lambda meta: (triton.cdiv(indices.shape[0], meta['BLOCK_Y']),)
    gather_kernel[grid](
        x0,
        indices,
        output,
        input_size=input_size,
        BLOCK_Y=16,
        BLOCK_X=32,
    )
    return output


def torch_func(x0: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    # 将输入展平，然后根据索引收集
    x_flat = x0.flatten()
    indices_flat = indices.flatten()
    # 确保索引在有效范围内
    indices_flat = torch.clamp(indices_flat, 0, x_flat.shape[0] - 1)
    gathered = x_flat[indices_flat]
    return gathered.reshape(x0.shape)


if __name__ == "__main__":
    torch.manual_seed(0)
    size = (16, 32)
    x = torch.randn(size, device="npu", dtype=torch.float32)
    # 生成随机索引，范围在 [0, x.numel())
    indices = torch.randint(0, x.numel(), size, device="npu", dtype=torch.int64)
    output_triton = triton_func(x, indices)
    output_torch = torch_func(x, indices)
    # 验证结果
    print(f"输入形状: x={x.shape}")
    print(f"索引形状: {indices.shape}")
    print(f"输出形状: {output_triton.shape}")
    print(f"最大误差: {torch.max(torch.abs(output_triton - output_torch)).item():.6f}")
    print(f"是否相等: {torch.allclose(output_triton, output_torch)}")

