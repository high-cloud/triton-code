import torch
import torch.npu
import triton
import triton.language as tl


@triton.jit
def select_kernel(
    condition_ptr,  # 条件张量的指针
    x_ptr,  # 输入张量 x 的指针
    y_ptr,  # 输入张量 y 的指针
    output_ptr,  # 输出张量的指针
    BLOCK_Y: tl.constexpr,
    BLOCK_X: tl.constexpr,
):
    pid = tl.program_id(0)  # 程序 ID
    
    blk_beg = pid * BLOCK_Y
    x1 = tl.arange(0, BLOCK_X)
    y0 = blk_beg + tl.arange(0, BLOCK_Y)
    idx = x1[None, :] + y0[:, None] * BLOCK_X
    
    # 加载条件、x 和 y
    cond = tl.load(condition_ptr + idx)
    tmp0 = tl.load(x_ptr + idx)
    tmp1 = tl.load(y_ptr + idx)
    
    # 使用 select 操作：如果条件为真选择 x，否则选择 y
    # 在 triton 中，select 通过 where 实现
    tmp2 = tl.where(cond, tmp0, tmp1)
    
    tl.store(output_ptr + idx, tmp2)


def triton_func(condition: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    assert condition.shape == x0.shape == x1.shape, "所有输入张量形状必须相同"
    output = torch.empty_like(x0)
    grid = lambda meta: (triton.cdiv(condition.shape[0], meta['BLOCK_Y']),)
    select_kernel[grid](
        condition,
        x0,
        x1,
        output,
        BLOCK_Y=16,
        BLOCK_X=32,
    )
    return output


def torch_func(condition: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    return torch.where(condition, x0, x1)


if __name__ == "__main__":
    torch.manual_seed(0)
    size = (16, 32)
    condition = torch.randn(size, device="npu", dtype=torch.float32) > 0
    x0 = torch.randn(size, device="npu", dtype=torch.float32)
    x1 = torch.randn(size, device="npu", dtype=torch.float32)
    output_triton = triton_func(condition, x0, x1)
    output_torch = torch_func(condition, x0, x1)
    # 验证结果
    print(f"条件形状: {condition.shape}")
    print(f"输入形状: x0={x0.shape}, x1={x1.shape}")
    print(f"输出形状: {output_triton.shape}")
    print(f"最大误差: {torch.max(torch.abs(output_triton - output_torch)).item():.6f}")
    print(f"是否相等: {torch.allclose(output_triton, output_torch)}")

