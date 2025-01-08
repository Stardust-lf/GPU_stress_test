import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import time
import subprocess
import csv


def set_gpu_frequency(offset):
    """
    Sets the GPU frequency offset using nvidia-settings.
    """
    command = f"nvidia-settings -a [gpu:0]/GPUGraphicsClockOffsetAllPerformanceLevels={offset}"
    subprocess.run(command, shell=True, check=True)


def get_gpu_stats():
    """
    Queries the current GPU frequency and power draw using nvidia-smi.
    Returns the frequency (MHz) and power draw (Watts).
    """
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
        writer.writerow(["Frequency (MHz)", "Power (Watts)", "Total Errors", "Test Duration (s)"])

        for freq in frequency_offsets:
            print(f"\nSetting GPU frequency offset to {freq}...")
            set_gpu_frequency(freq)
            time.sleep(5)

            a = np.random.randn(matrix_size, matrix_size).astype(np.float32)
            b = np.random.randn(matrix_size, matrix_size).astype(np.float32)

            a_gpu = cuda.mem_alloc(a.nbytes)
            b_gpu = cuda.mem_alloc(b.nbytes)
            c_gpu = cuda.mem_alloc(a.nbytes)
            c_expected_gpu = cuda.mem_alloc(a.nbytes)

            cuda.memcpy_htod(a_gpu, a)
            cuda.memcpy_htod(b_gpu, b)

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

            block_size = 32
            grid_size = (matrix_size + block_size - 1) // block_size

            func(a_gpu, b_gpu, c_expected_gpu, np.int32(matrix_size), block=(block_size, block_size, 1), grid=(grid_size, grid_size, 1))
            cuda.Context.synchronize()

            start_time = time.time()
            iteration = 0
            error_count = 0

            print(f"Running test for up to {test_duration} seconds or until {max_errors} errors at frequency offset {freq}...")

            while time.time() - start_time < test_duration:
                func(a_gpu, b_gpu, c_gpu, np.int32(matrix_size), block=(block_size, block_size, 1), grid=(grid_size, grid_size, 1))
                cuda.Context.synchronize()

                c_result = np.empty_like(a)
                c_expected = np.empty_like(a)
                cuda.memcpy_dtoh(c_result, c_gpu)
                cuda.memcpy_dtoh(c_expected, c_expected_gpu)

                if not np.allclose(c_result, c_expected, atol=1e-5):
                    error_count += 1
                    if error_count >= max_errors:
                        break

                iteration += 1

            test_duration_actual = time.time() - start_time
            frequency, power = get_gpu_stats()

            writer.writerow([frequency, power, error_count, test_duration_actual])
            print(f"Frequency offset {freq}: Total errors = {error_count}, Test duration = {test_duration_actual:.2f} seconds.")

    print("\nResetting GPU frequency offset to default...")
    set_gpu_frequency(0)

    print(f"Test completed. Results saved to {output_file}")


# Run the GPU stress test
gpu_stress_test(matrix_size=1024 * 8, max_errors=1000, test_duration=1800, frequency_offsets=list(range(350, 249, -5)), output_file="./data/8192_1800.csv")
