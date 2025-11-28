import torch
import torch.npu
import triton
import triton.language as tl


@triton.jit
def triton_kernel(
    input_ptr,  # 输入张量的指针 (4, 8, 64)，交错的张量
    output_ptr,  # 输出张量的指针 (4, 8, 32)
    channel_idx,  # 0 或 1，表示提取哪个通道（0=偶数位置，1=奇数位置）
    BLOCK_B: tl.constexpr,  # batch 维度的 block 大小
    BLOCK_H: tl.constexpr,  # height 维度大小
    BLOCK_W: tl.constexpr,  # width 维度的 block 大小
):
    # 固定维度大小
    B = 4
    H = 8
    W = 32
    W_input = 64
    
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

    # 输入索引：input[b, h, 2*w + channel_idx]
    # 对于 channel_idx=0，取偶数索引 (0, 2, 4, ...)
    # 对于 channel_idx=1，取奇数索引 (1, 3, 5, ...)
    input_w_idx = w_expanded * 2 + channel_idx  # [1, 1, BLOCK_W]
    
    input_idx = b_expanded * (H * W_input) + h_expanded * W_input + input_w_idx

    # 使用 extract_slice 的方式加载数据
    # 通过 stride=2 和 offset=channel_idx 来提取切片
    data = tl.load(input_ptr + input_idx)

    # 输出索引：output[b, h, w]
    output_idx = b_expanded * (H * W) + h_expanded * W + w_expanded

    # 存储结果
    tl.store(output_ptr + output_idx, data)


def triton_func(input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert len(input_tensor.shape) == 3, "输入必须是三维张量"
    B, H, W_input = input_tensor.shape
    
    # 固定维度大小，确保能被 block 大小整除
    assert B == 4, "B 维度必须为 4"
    assert H == 8, "H 维度必须为 8"
    assert W_input == 64, "输入 W 维度必须为 64"
    W = 32
    
    # 输出形状：(B, H, W)
    x = torch.empty((B, H, W), device=input_tensor.device, dtype=input_tensor.dtype)
    y = torch.empty((B, H, W), device=input_tensor.device, dtype=input_tensor.dtype)
    
    # 设置 block 大小（必须能整除对应维度）
    BLOCK_B = 4  # 必须能整除 B=4
    BLOCK_H = 8  # 必须能整除 H=8
    BLOCK_W = 32  # 必须能整除 W=32
    
    grid = lambda meta: (triton.cdiv(B, meta['BLOCK_B']),)
    
    # 提取 channel 0 (偶数位置 -> x)
    triton_kernel[grid](
        input_tensor,
        x,
        0,           # channel_idx = 0 (偶数位置)
        BLOCK_B=BLOCK_B,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
    )
    
    return x


def torch_func(input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # 使用 reshape 和 split 实现 deinterleave
    # reshape 成 (B, H, W, 2)，然后 split 成两个 (B, H, W)
    B, H, W_input = input_tensor.shape
    W = W_input // 2
    reshaped = input_tensor.reshape(B, H, W, 2)  # (B, H, W, 2)
    x, _ = reshaped.split(1, dim=3)  # 每个是 (B, H, W, 1)
    x = x.squeeze(3)  # (B, H, W)
    # y = y.squeeze(3)  # (B, H, W)
    return x


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

