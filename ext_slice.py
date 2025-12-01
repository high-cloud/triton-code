import torch
import torch.npu
import triton
import triton.language as tl


@triton.jit
def triton_kernel(
    input_ptr,
    output_ptr,
    w_size,
    BLOCK_H: tl.constexpr,  # height 维度大小
    BLOCK_W: tl.constexpr,  # width 维度的 block 大小
):
    # 固定维度大小
    H = 8
    W = 32
    W_input = 64
    
    pid = tl.program_id(0)

    # 计算当前 block 的起始位置
    h_idx = pid * BLOCK_H + tl.arange(0, BLOCK_H)
    w_idx = tl.arange(0, BLOCK_W)

    # 计算全局索引
    h_expanded = h_idx[:, None]  # [BLOCK_H, 1]
    w_expanded_base = w_idx[None, :]  # [1, BLOCK_W]

    num_iter = w_size // BLOCK_W // 2
    # 输入索引：input[b, h, 2*w + channel_idx]
    # 对于 channel_idx=0，取偶数索引 (0, 2, 4, ...)
    # 对于 channel_idx=1，取奇数索引 (1, 3, 5, ...)

    for i in range(num_iter):
        w_expanded = w_expanded_base + i * BLOCK_W  # [1, BLOCK_W]
        input_w_idx = w_expanded * 2 + channel_idx  # [1, BLOCK_W]
    
        input_idx = h_expanded * w_size + input_w_idx

        data = tl.load(input_ptr + input_idx)

        output_idx = b_expanded * (H * W) + h_expanded * W + w_expanded

        tl.store(output_ptr + output_idx, data)


def triton_func(input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert len(input_tensor.shape) == 2, "输入必须是三维张量"
    H, W_input = input_tensor.shape
    
    # 固定维度大小，确保能被 block 大小整除
    assert H == 8, "H 维度必须为 8"
    assert W_input == 128, "输入 W 维度必须为 64"
    W = 64

    # 输出形状：(H, W)
    x = torch.empty((H, W), device=input_tensor.device, dtype=input_tensor.dtype)
    
    # 设置 block 大小（必须能整除对应维度）
    BLOCK_H = 8  # 必须能整除 H=8
    BLOCK_W = 32  # 必须能整除 W=32
    
    grid = lambda meta: (triton.cdiv(H, meta['BLOCK_H']),)
    
    # 提取 channel 0 (偶数位置 -> x)
    triton_kernel[grid](
        input_tensor,
        x,
        W_input,           # channel_idx = 0 (偶数位置)
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
    )
    
    return x


def torch_func(input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # 使用 reshape 和 split 实现 deinterleave
    # reshape 成 (H, W, 2)，然后 split 成两个 (H, W)
    H, W_input = input_tensor.shape
    W = W_input // 2
    reshaped = input_tensor.reshape(H, W, 2)  # (H, W, 2)
    x, _ = reshaped.split(1, dim=2)  # 每个是 (H, W, 1)
    x = x.squeeze(2)  # (H, W)
    # y = y.squeeze(3)  # (H, W)
    return x


if __name__ == "__main__":
    torch.manual_seed(0)
    # 先创建两个原始张量并 interleave，然后测试 deinterleave
    size = (8, 128)  # (H, W) 二维张量
    input_tensor = torch.randn(size, device="npu", dtype=torch.float32)
    
    x_triton = triton_func(input_tensor)
    x_torch = torch_func(input_tensor)
    
    # 验证结果
    print(f"输入形状: input={input_tensor.shape}")
    print(f"输出形状: x={x_triton.shape}")
    print(f"x 最大误差: {torch.max(torch.abs(x_triton - x_torch)).item():.6f}")
    print(f"x 是否相等: {torch.allclose(x_triton, x_torch, rtol=1e-5, atol=1e-5)}")

