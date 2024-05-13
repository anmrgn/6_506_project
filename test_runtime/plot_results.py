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
    number_bits_start = 20
    number_bits_end = 30
    number_iterations = 50
    distribution = "mixed_normal"

    number_bits_array = np.array(range(number_bits_start, number_bits_end))

    results = []
    for algorithm in algorithms:
        results.append(load_range_number_bits(algorithm, distribution, number_bits_start, number_bits_end, number_iterations))

    for algorithm, result in zip(algorithms, results):
        plt.plot(number_bits_array, np.log(result))

    plt.legend(algorithms)
    plt.xlabel("Number bits")
    plt.ylabel("Log runtime")
    plt.show()

    serial_naive_result, serial_cache_oblivious_result, parallel_naive_result, parallel_cache_oblivious_result = results

    serial_speedup = serial_naive_result / serial_cache_oblivious_result
    parallel_speedup = parallel_naive_result / parallel_cache_oblivious_result

    plt.plot(number_bits_array, serial_speedup)
    plt.plot(number_bits_array, parallel_speedup)
    plt.plot(number_bits_array, np.array([1 for _ in number_bits_array]))
    plt.legend(["serial speedup", "parallel speedup", "y = 1"])
    plt.xlabel("Number bits")
    plt.ylabel("Speedup")
    plt.show()

if __name__ == "__main__":
    main()