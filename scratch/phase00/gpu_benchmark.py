import torch
import time

size = 5000

a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)

start = time.time()
c_cpu = a_cpu @ b_cpu
cpu_time = time.time() - start
print(f"CPU: {cpu_time:.3f}s")

if torch.backends.mps.is_available():
    a_gpu = a_cpu.to("mps")
    b_gpu = b_cpu.to("mps")

    # warm-up run: the first GPU run includes start-up cost
    _ = a_gpu @ b_gpu
    torch.mps.synchronize()

    start = time.time()
    c_gpu = a_gpu @ b_gpu
    torch.mps.synchronize()
    gpu_time = time.time() - start
    print(f"MPS: {gpu_time:.3f}s")
    print(f"Speedup: {cpu_time / gpu_time:.0f}x")
