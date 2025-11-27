import torch
import torch.npu
import triton
import triton.language as tl


@triton.jit
def deinterleave_slice_kernel(
    input_ptr,  # 输入张量的指针 (B, H, 2*W)，交错的张量
    output_ptr,  # 输出张量的指针 (B, H, W)
    input_shape_0,  # B
    input_shape_1,  # H
    input_shape_2,  # 2*W
    input_stride_0,  # H * 2*W
    input_stride_1,  # 2*W
    input_stride_2,  # 1
    channel_idx,  # 0 或 1，表示提取哪个通道（0=偶数位置，1=奇数位置）
    slice_size_0,  # B
    slice_size_1,  # H
    slice_size_2,  # W
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    indices = block_start + tl.arange(0, BLOCK_SIZE)

    # 计算输出索引 (B, H, W)
    idx_0 = indices // (slice_size_1 * slice_size_2)  # b
    idx_1 = (indices // slice_size_2) % slice_size_1  # h
    idx_2 = indices % slice_size_2  # w

    # 在输入中，最后一维的索引需要考虑步长 2 和通道索引
    # 对于 channel_idx=0，取偶数索引 (0, 2, 4, ...)
    # 对于 channel_idx=1，取奇数索引 (1, 3, 5, ...)
    input_idx_0 = idx_0
    input_idx_1 = idx_1
    input_idx_2 = idx_2 * 2 + channel_idx  # 关键：步长为 2，加上通道偏移

    # 掩码（无 mask，因为维度固定）
    # mask = (idx_0 < slice_size_0) & (idx_1 < slice_size_1) & (idx_2 < slice_size_2)

    # 计算输入指针偏移
    input_offsets = (
        input_idx_0 * input_stride_0 +
        input_idx_1 * input_stride_1 +
        input_idx_2 * input_stride_2
    )
    input_ptrs = input_ptr + input_offsets

    # 加载数据
    data = tl.load(input_ptrs)

    # 计算输出指针偏移
    output_offsets = (
        idx_0 * slice_size_1 * slice_size_2 +
        idx_1 * slice_size_2 +
        idx_2
    )
    output_ptrs = output_ptr + output_offsets
    tl.store(output_ptrs, data)


def triton_func(input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert len(input_tensor.shape) == 3, "输入必须是三维张量"
    B, H, W_input = input_tensor.shape
    assert W_input % 2 == 0, "输入的 W 维度必须是偶数"
    W = W_input // 2
    
    # 固定维度大小，确保能被 block 大小整除
    assert B == 4, "B 维度必须为 4"
    assert H == 8, "H 维度必须为 8"
    assert W == 32, "W 维度必须为 32（输入 W 必须是 64）"
    
    # 输出形状：(B, H, W)
    x = torch.empty((B, H, W), device=input_tensor.device, dtype=input_tensor.dtype)
    y = torch.empty((B, H, W), device=input_tensor.device, dtype=input_tensor.dtype)
    
    # 计算输入 stride
    input_stride_0 = H * W_input  # B 维度的 stride
    input_stride_1 = W_input      # H 维度的 stride
    input_stride_2 = 1             # W 维度的 stride
    
    # 设置 block 大小
    BLOCK_SIZE = 1024  # 一次处理 1024 个元素
    total_elements = B * H * W
    
    grid = lambda meta: (triton.cdiv(total_elements, meta['BLOCK_SIZE']),)
    
    # 提取 channel 0 (偶数位置 -> x)
    deinterleave_slice_kernel[grid](
        input_tensor,
        x,
        B,           # input_shape_0
        H,           # input_shape_1
        W_input,     # input_shape_2
        input_stride_0,
        input_stride_1,
        input_stride_2,
        0,           # channel_idx = 0 (偶数位置)
        B,           # slice_size_0
        H,           # slice_size_1
        W,           # slice_size_2
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # 提取 channel 1 (奇数位置 -> y)
    deinterleave_slice_kernel[grid](
        input_tensor,
        y,
        B,           # input_shape_0
        H,           # input_shape_1
        W_input,     # input_shape_2
        input_stride_0,
        input_stride_1,
        input_stride_2,
        1,           # channel_idx = 1 (奇数位置)
        B,           # slice_size_0
        H,           # slice_size_1
        W,           # slice_size_2
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return x, y


def torch_func(input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # 使用 reshape 和 split 实现 deinterleave
    # reshape 成 (B, H, W, 2)，然后 split 成两个 (B, H, W)
    B, H, W_input = input_tensor.shape
    W = W_input // 2
    reshaped = input_tensor.reshape(B, H, W, 2)  # (B, H, W, 2)
    x, y = reshaped.split(1, dim=3)  # 每个是 (B, H, W, 1)
    x = x.squeeze(3)  # (B, H, W)
    y = y.squeeze(3)  # (B, H, W)
    return x, y


if __name__ == "__main__":
    torch.manual_seed(0)
    # 先创建两个原始张量并 interleave，然后测试 deinterleave
    size = (4, 8, 32)  # (B, H, W) 三维张量
    x_orig = torch.randn(size, device="npu", dtype=torch.float32)
    y_orig = torch.randn(size, device="npu", dtype=torch.float32)
    
    # 创建交错的输入（模拟 interleave 的结果）
    stacked = torch.stack([x_orig, y_orig], dim=0)  # (2, B, H, W)
    input_tensor = stacked.permute(1, 2, 3, 0).reshape(4, 8, 64)  # (B, H, 2*W)
    
    x_triton, y_triton = triton_func(input_tensor)
    x_torch, y_torch = torch_func(input_tensor)
    
    # 验证结果
    print(f"输入形状: input={input_tensor.shape}")
    print(f"输出形状: x={x_triton.shape}, y={y_triton.shape}")
    print(f"x 最大误差: {torch.max(torch.abs(x_triton - x_torch)).item():.6f}")
    print(f"y 最大误差: {torch.max(torch.abs(y_triton - y_torch)).item():.6f}")
    print(f"x 是否相等: {torch.allclose(x_triton, x_torch, rtol=1e-5, atol=1e-5)}")
    print(f"y 是否相等: {torch.allclose(y_triton, y_torch, rtol=1e-5, atol=1e-5)}")
    print(f"\n原始 x 示例 (前2个batch, 前2个H, 前4个W):\n{x_orig[:2, :2, :4]}")
    print(f"\n原始 y 示例 (前2个batch, 前2个H, 前4个W):\n{y_orig[:2, :2, :4]}")
    print(f"\nTriton 输出 x 示例 (前2个batch, 前2个H, 前4个W):\n{x_triton[:2, :2, :4]}")
    print(f"\nTriton 输出 y 示例 (前2个batch, 前2个H, 前4个W):\n{y_triton[:2, :2, :4]}")
    print(f"\n输入交错张量示例 (前2个batch, 前2个H, 前8个W):\n{input_tensor[:2, :2, :8]}")

