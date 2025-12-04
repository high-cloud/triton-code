import torch
import torch.npu
import triton
import triton.language as tl


@triton.jit
def triton_kernel(
    input_ptr,
    output_ptr,
    w_size,
    BLOCK_B: tl.constexpr,  # batch 维度的 block 大小
    BLOCK_H: tl.constexpr,  # height 维度大小
    BLOCK_W: tl.constexpr,  # width 维度的 block 大小
):
    pid = tl.program_id(0)
    w_size = 128

    # 计算当前 block 的起始位置
    b_idx = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    h_idx = tl.arange(0, BLOCK_H)
    w_idx = tl.arange(0, BLOCK_W)

    # 计算全局索引
    b_expanded = b_idx[:, None, None]  # [BLOCK_B, 1, 1]
    h_expanded = h_idx[None, :, None]  # [1, BLOCK_H, 1]
    w_expanded_base = w_idx[None, None, :]  # [1, 1, BLOCK_W]

    num_iter = w_size // BLOCK_W // 2
    w_output_size = w_size // 2
    for i in range(num_iter):
        w_expanded = w_expanded_base + i * BLOCK_W  # [1, BLOCK_W]
        input_w_idx = w_expanded * 2 # [1, BLOCK_W]
    
        input_idx = b_expanded * (BLOCK_H * w_size) + h_expanded * w_size + input_w_idx

        data = tl.load(input_ptr + input_idx)

        output_idx = b_expanded * (BLOCK_H * w_output_size) + h_expanded * (w_output_size) + w_expanded

        tl.store(output_ptr + output_idx, data)


def triton_func(input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert len(input_tensor.shape) == 3, "输入必须是三维张量"
    B, H, W_input = input_tensor.shape
    
    # 固定维度大小，确保能被 block 大小整除
    assert B == 4, "B 维度必须为 4"
    assert H == 8, "H 维度必须为 8"
    assert W_input == 128, "输入 W 维度必须为 128"
    W = 64

    # 输出形状：(B, H, W)
    x = torch.empty((B, H, W), device=input_tensor.device, dtype=input_tensor.dtype)

    # 设置 block 大小（必须能整除对应维度）
    BLOCK_B = 4  # 必须能整除 B=4
    BLOCK_H = 8  # 必须能整除 H=8
    BLOCK_W = 32  # 必须能整除 W=32

    grid = lambda meta: (triton.cdiv(B, meta['BLOCK_B']),)

    # 提取 channel 0 (偶数位置 -> x)
    triton_kernel[grid](
        input_tensor,
        x,
        W_input,
        BLOCK_B=BLOCK_B,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
    )

    return x


def torch_func(input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # 使用 reshape 和 split 实现 deinterleave
    # reshape 成 (H, W, 2)，然后 split 成两个 (H, W)
    B, H, W_input = input_tensor.shape
    W = W_input // 2
    reshaped = input_tensor.reshape(B, H, W, 2)  # (B, H, W, 2)
    x, _ = reshaped.split(1, dim=3)  # 每个是 (B, H, W, 1)
    x = x.squeeze(3)  # (B, H, W)
    return x


if __name__ == "__main__":
    torch.manual_seed(0)
    # 先创建两个原始张量并 interleave，然后测试 deinterleave
    size = (4, 8, 128)  # (B, H, W) 三维张量
    input_tensor = torch.randn(size, device="npu", dtype=torch.float32)
    
    x_triton = triton_func(input_tensor)
    x_torch = torch_func(input_tensor)
    
    # 验证结果
    print(f"输入形状: input={input_tensor.shape}")
    print(f"输出形状: x={x_triton.shape}")
    print(f"x 最大误差: {torch.max(torch.abs(x_triton - x_torch)).item():.6f}")
    print(f"x 是否相等: {torch.allclose(x_triton, x_torch, rtol=1e-5, atol=1e-5)}")

