import os
import subprocess
import numpy as np
import time
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

def set_gpu_frequency(offset):
    """Set GPU clock offset using nvidia-settings."""
    command = f"nvidia-settings -a [gpu:0]/GPUGraphicsClockOffsetAllPerformanceLevels={offset}"
    subprocess.run(command, shell=True, check=True)

def gpu_stress_test(matrix_size=1024, duration=60):
    """Run GPU stress test and calculate error epochs."""
    # Create random matrices
    a = np.random.randn(matrix_size, matrix_size).astype(np.float32)
    b = np.random.randn(matrix_size, matrix_size).astype(np.float32)

    # Allocate GPU memory
    a_gpu = cuda.mem_alloc(a.nbytes)
    b_gpu = cuda.mem_alloc(b.nbytes)
    c_gpu = cuda.mem_alloc(a.nbytes)  # Result matrix

    # Copy matrices to GPU
    cuda.memcpy_htod(a_gpu, a)
    cuda.memcpy_htod(b_gpu, b)

    # CUDA kernel
    mod = SourceModule(
        """
        __global__ void matrix_multiply(float *a, float *b, float *c, int matrix_size) {
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;
            if (row < matrix_size && col < matrix_size) {
                float value = 0;
                for (int k = 0; k < matrix_size; ++k) {
                    value += a[row * matrix_size + k] * b[k * matrix_size + col];
                }
                c[row * matrix_size + col] = value;
            }
        }
        """
    )
    func = mod.get_function("matrix_multiply")

    block_size = 32
    grid_size = (matrix_size + block_size - 1) // block_size

    # Run test
    start_time = time.time()
    error_epochs = 0

    while time.time() - start_time < duration:
        func(a_gpu, b_gpu, c_gpu, np.int32(matrix_size), block=(block_size, block_size, 1), grid=(grid_size, grid_size, 1))
        cuda.Context.synchronize()

        # Validate
        c_result = np.empty_like(a)
        cuda.memcpy_dtoh(c_result, c_gpu)
        expected = np.dot(a, b)
        if not np.array_equal(c_result, expected):
            error_epochs += 1

    return error_epochs

def main():
    target_frequencies = list(range(360, 270, -10))
    results = {}

    for freq in target_frequencies:
        set_gpu_frequency(freq)
        print(f"Running test at frequency offset: {freq} MHz")

        # Collect errors until we reach 3
        errors = []
        while len(errors) < 3:
            error_epochs = gpu_stress_test(matrix_size=1024*4, duration=60)
            errors.append(error_epochs)
            print(f"Frequency {freq} MHz, Error Epochs: {error_epochs}")

        avg_errors = np.mean(errors)
        min_errors = np.min(errors)
        results[freq] = (min_errors, avg_errors)
        print(f"Frequency {freq} MHz completed. Min Error Epochs: {min_errors}, Average Error Epochs: {avg_errors}\n")

    # Final report
    print("Final Report:")
    for freq, (min_errors, avg_errors) in results.items():
        print(f"Frequency: {freq} MHz, Min Error Epochs: {min_errors}, Average Error Epochs: {avg_errors}")

if __name__ == "__main__":
    main()
