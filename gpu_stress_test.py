import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import time
import subprocess
import csv


def set_gpu_frequency(offset):
    command = f"nvidia-settings -a [gpu:0]/GPUGraphicsClockOffsetAllPerformanceLevels={offset}"
    subprocess.run(command, shell=True, check=True)


def get_gpu_stats():
    command = "nvidia-smi --query-gpu=clocks.gr,power.draw --format=csv,noheader,nounits"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError("Failed to query GPU stats with nvidia-smi.")
    frequency, power = result.stdout.strip().split(',')
    return frequency.strip(), power.strip()


def gpu_stress_test(matrix_size=1024, max_errors=20, test_duration=180, frequency_offsets=None, output_file="gpu_stress_test.csv"):
    if frequency_offsets is None:
        frequency_offsets = list(range(350, 269, -5))

    with open(output_file, mode="w+", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "Frequency (MHz)", 
            "Power (Watts)", 
            "Error Time (s)", 
            "Error Distance", 
            "Hamming Distance", 
            "Error Iterations Since Last Error"
        ])

        for freq in frequency_offsets:
            print(f"\nSetting GPU frequency offset to {freq}...")
            set_gpu_frequency(freq)
            time.sleep(2)

            # Initialize data and allocate GPU memory
            a = np.random.randint(0, np.iinfo(np.uint64).max, (matrix_size, matrix_size), dtype=np.uint64)
            b = np.random.randint(0, np.iinfo(np.uint64).max, (matrix_size, matrix_size), dtype=np.uint64)
            a_gpu = cuda.mem_alloc(a.nbytes)
            b_gpu = cuda.mem_alloc(b.nbytes)
            c_gpu = cuda.mem_alloc(a.nbytes)
            c_expected_gpu = cuda.mem_alloc(a.nbytes)

            cuda.memcpy_htod(a_gpu, a)
            cuda.memcpy_htod(b_gpu, b)

            # Compile and get the matrix multiplication kernel
            mod = SourceModule(
                """
                __global__ void matrix_multiply(unsigned *a, unsigned *b, unsigned *c, int matrix_size)
                {
                    int row = blockIdx.y * blockDim.y + threadIdx.y;
                    int col = blockIdx.x * blockDim.x + threadIdx.x;
                    if (row < matrix_size && col < matrix_size) {
                        unsigned value = 0;
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

            # Compute expected results
            func(a_gpu, b_gpu, c_expected_gpu, np.int32(matrix_size), block=(block_size, block_size, 1), grid=(grid_size, grid_size, 1))
            cuda.Context.synchronize()

            # Initialize testing variables
            start_time = time.perf_counter()
            last_write_time = start_time
            last_error_iteration = 0
            iteration = 0
            error_count = 0

            print(f"Running test for up to {test_duration} seconds or until {max_errors} errors at frequency offset {freq}...")

            while time.perf_counter() - start_time < test_duration:
                # Execute the kernel
                func(a_gpu, b_gpu, c_gpu, np.int32(matrix_size), block=(block_size, block_size, 1), grid=(grid_size, grid_size, 1))
                cuda.Context.synchronize()

                # Copy results back to host and compare
                c_result = np.empty_like(a)
                c_expected = np.empty_like(a)
                cuda.memcpy_dtoh(c_result, c_gpu)
                cuda.memcpy_dtoh(c_expected, c_expected_gpu)

                if not np.array_equal(c_result, c_expected):
                    current_time = time.perf_counter()
                    error_time_since_last_write = current_time - last_write_time
                    error_iterations_since_last_error = iteration - last_error_iteration

                    # Compute error distance and Hamming distance
                    error_distance = np.sum(c_result - c_expected)
                    error_mask = c_result != c_expected
                    hamming_distance = np.sum(error_mask)

                    # Get GPU stats and write to CSV
                    frequency, power = get_gpu_stats()
                    writer.writerow([
                        frequency, 
                        power, 
                        error_time_since_last_write, 
                        error_distance, 
                        hamming_distance, 
                        error_iterations_since_last_error
                    ])

                    # Update error tracking variables
                    error_count += 1
                    last_write_time = current_time
                    last_error_iteration = iteration

                    if error_count >= max_errors:
                        break

                iteration += 1

            print(f"Frequency offset {freq}: Total errors = {error_count}.")

            if error_count >= max_errors:
                print(f"Maximum errors reached at frequency {freq}. Moving to the next frequency.")

    print("\nResetting GPU frequency offset to default...")
    set_gpu_frequency(0)

    print(f"Test completed. Results saved to {output_file}")
gpu_stress_test(matrix_size=1024 * 2, max_errors=1000, test_duration=1800, frequency_offsets=list(range(300, -1, -15)), output_file="./data/result.csv")
