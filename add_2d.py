import torch
import triton
import triton.language as tl


@triton.jit
def add_2d_kernel(
    x_ptr,  # 输入张量 x 的指针
    y_ptr,  # 输入张量 y 的指针
    output_ptr,  # 输出张量的指针
    M,  # 行数
    N,  # 列数
    stride_xm, stride_xn,  # x 的步长
    stride_ym, stride_yn,  # y 的步长
    stride_om, stride_on,  # output 的步长
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    2D 张量加法 kernel
    
    参数:
        x_ptr, y_ptr: 输入张量的指针
        output_ptr: 输出张量的指针
        M, N: 张量的维度 (M x N)
        stride_*: 各张量的内存步长
        BLOCK_SIZE_M, BLOCK_SIZE_N: 每个 block 处理的行数和列数
    """
    # 获取当前程序 ID
    pid_m = tl.program_id(0)  # 行方向的程序 ID
    pid_n = tl.program_id(1)  # 列方向的程序 ID
    
    # 计算当前 block 要处理的行和列的起始位置
    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N
    
    # 创建偏移量，用于访问当前 block 内的元素
    offsets_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    
    # 创建 2D 偏移量网格
    offsets = offsets_m[:, None] * stride_xm + offsets_n[None, :] * stride_xn
    
    # 创建掩码，确保不越界
    mask = (offsets_m[:, None] < M) & (offsets_n[None, :] < N)
    
    # 从全局内存加载数据到寄存器
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # 执行加法运算
    output = x + y
    
    # 计算输出张量的偏移量
    output_offsets = offsets_m[:, None] * stride_om + offsets_n[None, :] * stride_on
    
    # 将结果写回全局内存
    tl.store(output_ptr + output_offsets, output, mask=mask)


def add_2d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    2D 张量加法的 Python 包装函数
    
    参数:
        x, y: 输入张量，形状为 (M, N)
    
    返回:
        output: 输出张量，形状为 (M, N)，其中 output[i, j] = x[i, j] + y[i, j]
    """
    # 检查输入形状
    assert x.shape == y.shape, f"输入张量形状必须相同: x.shape={x.shape}, y.shape={y.shape}"
    assert x.dim() == 2, f"输入必须是 2D 张量，但得到 {x.dim()}D"
    
    M, N = x.shape
    
    # 分配输出张量
    output = torch.empty_like(x)
    
    # 设置 block 大小
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    
    # 计算 grid 大小（需要多少个 block）
    grid = (
        triton.cdiv(M, BLOCK_SIZE_M),  # 行方向的 block 数量
        triton.cdiv(N, BLOCK_SIZE_N),  # 列方向的 block 数量
    )
    
    # 启动 kernel
    add_2d_kernel[grid](
        x,
        y,
        output,
        M,
        N,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return output


# 测试代码
if __name__ == "__main__":
    # 设置随机种子以便复现
    torch.manual_seed(0)
    
    # 创建测试张量
    M, N = 1024, 512
    x = torch.randn((M, N), device="cuda", dtype=torch.float32)
    y = torch.randn((M, N), device="cuda", dtype=torch.float32)
    
    # 使用 Triton kernel 计算
    output_triton = add_2d(x, y)
    
    # 使用 PyTorch 原生实现验证
    output_torch = x + y
    
    # 验证结果
    print(f"输入形状: x={x.shape}, y={y.shape}")
    print(f"输出形状: {output_triton.shape}")
    print(f"最大误差: {torch.max(torch.abs(output_triton - output_torch)).item():.6f}")
    print(f"是否相等: {torch.allclose(output_triton, output_torch)}")
    
    # 性能测试
    import time
    
    # 预热
    for _ in range(10):
        _ = add_2d(x, y)
    
    # 测试 Triton kernel
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = add_2d(x, y)
    torch.cuda.synchronize()
    triton_time = (time.time() - start) / 100
    
    # 测试 PyTorch
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = x + y
    torch.cuda.synchronize()
    torch_time = (time.time() - start) / 100
    
    print(f"\n性能对比:")
    print(f"Triton kernel: {triton_time*1000:.4f} ms")
    print(f"PyTorch: {torch_time*1000:.4f} ms")
    print(f"加速比: {torch_time/triton_time:.2f}x")

