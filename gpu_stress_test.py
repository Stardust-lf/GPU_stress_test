import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import time
import subprocess
import csv


def set_gpu_frequency(offset):
    """
    Sets the GPU frequency offset using `nvidia-settings`.
    """
    command = f"nvidia-settings -a [gpu:0]/GPUGraphicsClockOffsetAllPerformanceLevels={offset}"
    subprocess.run(command, shell=True, check=True)


def get_gpu_frequency_and_power():
    """
    Queries the current GPU frequency and power draw using `nvidia-smi`.
    Returns the frequency (MHz) and power draw (Watts).
    """
    command = "nvidia-smi --query-gpu=clocks.gr,power.draw --format=csv,noheader,nounits"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError("Failed to query GPU stats with nvidia-smi.")
    frequency, power = result.stdout.strip().split(',')
    return frequency.strip(), power.strip()


def gpu_stress_test(matrix_size=1024, max_errors=20, test_duration=180, frequency_offsets=None, output_file="gpu_error_log.csv"):
    """
    GPU stress test that switches frequency offsets and measures error statistics.
    Logs errors to a CSV file.
    """
    if frequency_offsets is None:
        frequency_offsets = list(range(320, 249, -5))  # Default: 320 to 250 decrementing by 5

    # Prepare CSV file for logging
    with open(output_file, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write the header
        writer.writerow(["Frequency (MHz)", "Power (Watts)", "Error Time (s)", "Error Magnitude"])

        for freq in frequency_offsets:
            print(f"\nSetting GPU frequency offset to {freq}...")
            set_gpu_frequency(freq)
            time.sleep(5)  # Allow GPU to stabilize
            
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
            
            # Test statistics
            start_time = time.time()
            last_error_time = start_time
            iteration = 0
            error_count = 0
            
            print(f"Running test for up to {test_duration} seconds or until {max_errors} errors at frequency offset {freq}...")
            
            while time.time() - start_time < test_duration:
                func(a_gpu, b_gpu, c_gpu, np.int32(matrix_size), block=(block_size, block_size, 1), grid=(grid_size, grid_size, 1))
                cuda.Context.synchronize()
                
                # Verify correctness after each iteration
                c_result = np.empty_like(a)
                c_expected = np.empty_like(a)
                cuda.memcpy_dtoh(c_result, c_gpu)
                cuda.memcpy_dtoh(c_expected, c_expected_gpu)
                
                if not np.allclose(c_result, c_expected, atol=1e-5):
                    current_time = time.time()
                    error_time_elapsed = current_time - last_error_time
                    frequency, power = get_gpu_frequency_and_power()
                    error_magnitude = np.abs(c_result - c_expected).max()
                    writer.writerow([frequency, power, error_time_elapsed, error_magnitude])
                    error_count += 1
                    print(f"Error {error_count} detected at iteration {iteration}.")
                    print(f"Time since last error: {error_time_elapsed:.2f} seconds.")
                    print(f"Current GPU frequency: {frequency} MHz, Power Draw: {power} W.")
                    print(f"Error Magnitude: {error_magnitude:.6f}")
                    last_error_time = current_time
                    
                    # Stop if maximum errors reached
                    if error_count >= max_errors:
                        break
                
                iteration += 1
            
            print(f"Frequency offset {freq}: Total errors = {error_count}.")
            
            # Stop early if maximum errors reached
            if error_count >= max_errors:
                print(f"Maximum errors reached at frequency {freq}. Moving to the next frequency.")
    
    # Reset GPU frequency offset to default
    print("\nResetting GPU frequency offset to default...")
    set_gpu_frequency(0)
    
    print(f"Test completed. Results saved to {output_file}")


# Run the GPU stress test
gpu_stress_test(matrix_size=1024, max_errors=60, test_duration=180, frequency_offsets=list(range(320, 279, -5)))
