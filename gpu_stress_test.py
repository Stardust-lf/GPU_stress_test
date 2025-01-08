import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import time
import subprocess
import csv
import argparse


def set_gpu_frequency(offset):
    """
    Sets the GPU frequency offset using `nvidia-settings`.
    """
    command = f"nvidia-settings -a [gpu:0]/GPUGraphicsClockOffsetAllPerformanceLevels={offset}"
    subprocess.run(command, shell=True, check=True)


def get_gpu_metrics():
    """
    Queries the GPU frequency, power draw, and SM utilization using `nvidia-smi`.
    Returns frequency (MHz), power draw (Watts), and SM utilization (%).
    """
    command = "nvidia-smi --query-gpu=clocks.gr,power.draw,utilization.gpu --format=csv,noheader,nounits"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError("Failed to query GPU stats with nvidia-smi.")
    frequency, power, sm_util = result.stdout.strip().split(',')
    return frequency.strip(), power.strip(), sm_util.strip()


def calculate_hamming_distance(a, b):
    """
    Calculate the Hamming distance between two binary matrices.
    """
    a_bin = np.unpackbits(a.view(np.uint8))
    b_bin = np.unpackbits(b.view(np.uint8))
    return np.sum(a_bin != b_bin)


def gpu_stress_test(matrix_size=256, max_errors=100, test_duration=180, freq_start=100, freq_end=0, freq_stride=5, output_file="gpu_stress_test.csv"):
    """
    GPU stress test that switches frequency offsets and measures error statistics.
    Logs errors, GPU metrics, and SM usage to a CSV file.
    """
    frequency_offsets = list(range(freq_start, freq_end - 1, -freq_stride))
    
    with open(output_file, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Frequency (MHz)", "Power (Watts)", "SM Usage (%)", "Error Time (s)", "Error Magnitude", "Hamming Distance"])

        for freq in frequency_offsets:
            print(f"\nSetting GPU frequency offset to {freq}...")
            set_gpu_frequency(freq)
            time.sleep(5)
            
            a = np.random.randint(0, 1000, (matrix_size, matrix_size)).astype(np.int32)
            b = np.random.randint(0, 1000, (matrix_size, matrix_size)).astype(np.int32)
            
            a_gpu = cuda.mem_alloc(a.nbytes)
            b_gpu = cuda.mem_alloc(b.nbytes)
            c_gpu = cuda.mem_alloc(a.nbytes)
            c_expected_gpu = cuda.mem_alloc(a.nbytes)
            
            cuda.memcpy_htod(a_gpu, a)
            cuda.memcpy_htod(b_gpu, b)
            
            mod = SourceModule(
                """
                #define PRIME 2147483647

                __global__ void matrix_multiply(int *a, int *b, int *c, int matrix_size) {
                    int row = blockIdx.y * blockDim.y + threadIdx.y;
                    int col = blockIdx.x * blockDim.x + threadIdx.x;
                    if (row < matrix_size && col < matrix_size) {
                        int value = 0;
                        for (int k = 0; k < matrix_size; ++k) {
                            value = (value + ((a[row * matrix_size + k] % PRIME) * (b[k * matrix_size + col] % PRIME)) % PRIME) % PRIME;
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
            last_error_time = start_time
            iteration = 0
            error_count = 0
            
            print(f"Running test for {test_duration} seconds or up to {max_errors} errors at frequency offset {freq}...")
            
            while time.time() - start_time < test_duration:
                func(a_gpu, b_gpu, c_gpu, np.int32(matrix_size), block=(block_size, block_size, 1), grid=(grid_size, grid_size, 1))
                cuda.Context.synchronize()
                
                c_result = np.empty_like(a)
                c_expected = np.empty_like(a)
                cuda.memcpy_dtoh(c_result, c_gpu)
                cuda.memcpy_dtoh(c_expected, c_expected_gpu)
                
                if not np.array_equal(c_result, c_expected):
                    current_time = time.time()
                    error_time_elapsed = current_time - last_error_time
                    frequency, power, sm_util = get_gpu_metrics()
                    error_magnitude = np.abs(c_result - c_expected).max()
                    hamming_distance = calculate_hamming_distance(c_result, c_expected)
                    writer.writerow([frequency, power, sm_util, error_time_elapsed, error_magnitude, hamming_distance])
                    error_count += 1
                    print(f"Error {error_count} detected. Time since last error: {error_time_elapsed:.2f}s.")
                    print(f"SM Usage: {sm_util}%, Frequency: {frequency} MHz, Power: {power} W.")
                    print(f"Error Magnitude: {error_magnitude}")
                    print(f"Hamming Distance: {hamming_distance}")
                    last_error_time = current_time
                    
                    if error_count >= max_errors:
                        break
                
                iteration += 1
            
            print(f"Frequency offset {freq}: Total errors = {error_count}.")
            
            if error_count >= max_errors:
                break
    
    print("\nResetting GPU frequency offset to default...")
    set_gpu_frequency(0)
    print(f"Test completed. Results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU Stress Test with SM Usage")
    parser.add_argument("-ms", "--matrix_size", type=int, default=256, help="Size of the square matrix.")
    parser.add_argument("-me", "--max_errors", type=int, default=100, help="Maximum number of errors before stopping.")
    parser.add_argument("-td", "--test_duration", type=int, default=180, help="Duration of the test in seconds.")
    parser.add_argument("-fs", "--freq_start", type=int, default=100, help="Starting frequency offset.")
    parser.add_argument("-fe", "--freq_end", type=int, default=0, help="Ending frequency offset.")
    parser.add_argument("-fstride", "--freq_stride", type=int, default=5, help="Frequency offset stride.")
    parser.add_argument("-of", "--output_file", type=str, default="gpu_stress_test.csv", help="Output CSV file for results.")
    
    args = parser.parse_args()
    print(args)
    gpu_stress_test(
        matrix_size=args.matrix_size,
        max_errors=args.max_errors,
        test_duration=args.test_duration,
        freq_start=args.freq_start,
        freq_end=args.freq_end,
        freq_stride=args.freq_stride,
        output_file=args.output_file,
    )
