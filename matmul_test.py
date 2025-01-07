
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import time
def gpu_stress_test(matrix_size=1024, duration=600):
    # Create two large random matrices
    a = np.random.randn(matrix_size, matrix_size).astype(np.float32)
    b = np.random.randn(matrix_size, matrix_size).astype(np.float32)
    # Allocate GPU memory for both matrices and result
    a_gpu = cuda.mem_alloc(a.nbytes)
    b_gpu = cuda.mem_alloc(b.nbytes)
    c_gpu = cuda.mem_alloc(a.nbytes)  # Result matrix
    c_expected_gpu = cuda.mem_alloc(a.nbytes)  # GPU-computed expected matrix
    # Copy matrices to GPU
    cuda.memcpy_htod(a_gpu, a)
    cuda.memcpy_htod(b_gpu, b)
    # Compile the CUDA kernel for matrix multiplication
    mod = SourceModule(
        """
        __global__ void matrix_multiply(float *a, float *b, float *c, int matrix_size)
        {
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
    # Define block and grid sizes
    block_size = 32  # Threads per block
    grid_size = (matrix_size + block_size - 1) // block_size
    # Compute the expected result once on GPU
    func(a_gpu, b_gpu, c_expected_gpu, np.int32(matrix_size), block=(block_size, block_size, 1), grid=(grid_size, grid_size, 1))
    cuda.Context.synchronize()
    # Record the start time
    start_time = time.time()
    iteration = 0
    # Perform the computation multiple times until the duration is reached
    while time.time() - start_time < duration:
        func(a_gpu, b_gpu, c_gpu, np.int32(matrix_size), block=(block_size, block_size, 1), grid=(grid_size, grid_size, 1))
        cuda.Context.synchronize()
        # Verify correctness after each iteration
        c_result = np.empty_like(a)
        c_expected = np.empty_like(a)
        cuda.memcpy_dtoh(c_result, c_gpu)
        cuda.memcpy_dtoh(c_expected, c_expected_gpu)
        if not np.allclose(c_result, c_expected, atol=1e-5):
            print(f"Error detected at iteration {iteration}.")
            print(f"Expected:\n{c_expected}\n")
            print(f"GPU Result:\n{c_result}\n")
            return
        iteration += 1
    # Record the end time
    end_time = time.time()
    # Print results and timing information
    print(f"Computation completed in {end_time - start_time:.2f} seconds.")
    print(f"Matrix size: {matrix_size} x {matrix_size}, Total iterations: {iteration}")
    print("All iterations passed correctness check.")
# Run the GPU stress test
gpu_stress_test(matrix_size=1024, duration=600)
