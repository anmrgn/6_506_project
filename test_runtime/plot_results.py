import os
import matplotlib.pyplot as plt
import numpy as np

TEST_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = "results"
EXE_PATH = os.path.abspath(os.path.join(TEST_DIR_PATH, "../bin/main"))

def load_single_file(algorithm, distribution, number_bits, number_iterations):
    file_name = f"{algorithm}_{distribution}_{number_bits}_{number_iterations}"
    file_path = os.path.join(TEST_DIR_PATH, RESULTS_DIR, file_name)

    rval = []
    with open(file_path, "r") as f:
        while True:
            time_ns_str = f.readline().strip()
            try:
                time_ns = int(time_ns_str)
                rval.append(time_ns)
            except ValueError as error:
                break
    return np.array(rval)

def load_range_number_bits(algorithm, distribution, number_bits_start, number_bits_end, number_iterations):
    rval = []
    for number_bits in range(number_bits_start, number_bits_end):
        curr = load_single_file(algorithm, distribution, number_bits, number_iterations)
        rval.append(np.mean(curr))
    return np.array(rval)

def main():
    algorithms = ["serial_naive", "serial_cache_oblivious", "parallel_naive", "parallel_cache_oblivious"]
    number_bits_start = 1
    number_bits_end = 28
    number_iterations = 50
    distribution = "uniform"
    distribution_to_name = {"uniform": "Uniform", "mixed_uniform": "Mixed Uniform", "standard_normal": "Standard Normal", "mixed_normal": "Mixed Normal"}
    distribution_name = distribution_to_name[distribution]

    number_bits_array = np.array(range(number_bits_start, number_bits_end))

    results = []
    for algorithm in algorithms:
        results.append(load_range_number_bits(algorithm, distribution, number_bits_start, number_bits_end, number_iterations))

    # for algorithm, result in zip(algorithms, results):
    #     plt.plot(number_bits_array, np.log(result))

    # plt.legend(algorithms)
    # plt.xlabel("Number bits")
    # plt.ylabel("Log runtime")
    # plt.show()

    serial_naive_result, serial_cache_oblivious_result, parallel_naive_result, parallel_cache_oblivious_result = results

    serial_speedup = serial_naive_result / serial_cache_oblivious_result
    parallel_speedup = parallel_naive_result / parallel_cache_oblivious_result

    plt.figure(figsize=(12, 9))
    plt.plot(number_bits_array, serial_speedup, linestyle='-', marker='o', markersize=8, color='blue')
    plt.title(f"Serial Speedup as a Function of Problem Size for {distribution_name} Distribution", fontsize=14, fontweight='bold', color='red', pad=20)
    plt.xlabel("Number bits (i.e., problem size is n = 2^b)", fontsize=12, fontweight='bold', color='blue', labelpad=10)
    plt.ylabel("Speedup", fontsize=12, fontweight='bold', color='green', labelpad=10)
    plt.xticks(fontsize=10, color='black')
    plt.yticks(fontsize=10, color='black')
    plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5, axis='both')
    plt.tight_layout()
    plt.savefig(f'serial_speedup_{distribution}', dpi=300)
    plt.show()
    
    plt.figure(figsize=(12, 9))
    plt.plot(number_bits_array, parallel_speedup, linestyle='-', marker='o', markersize=8, color='blue')
    plt.title(f"Parallel Speedup as a Function of Problem Size for {distribution_name} Distribution", fontsize=14, fontweight='bold', color='red', pad=20)
    plt.xlabel("Number bits (i.e., problem size is n = 2^b)", fontsize=12, fontweight='bold', color='blue', labelpad=10)
    plt.ylabel("Speedup", fontsize=12, fontweight='bold', color='green', labelpad=10)
    plt.xticks(fontsize=10, color='black')
    plt.yticks(fontsize=10, color='black')
    plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5, axis='both')
    plt.tight_layout()
    plt.savefig(f'parallel_speedup_{distribution}', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()