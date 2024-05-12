import os

TEST_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = "results"
EXE_PATH = os.path.abspath(os.path.join(TEST_DIR_PATH, "../bin/main"))

def trial(algorithm, distribution, number_bits, number_iterations):
    out_file_path = os.path.join(TEST_DIR_PATH, RESULTS_DIR, f"{algorithm}_{distribution}_{number_bits}_{number_iterations}")
    os.system(f"{EXE_PATH} {algorithm} {distribution} {number_bits} {number_iterations} >> {out_file_path}")

def fixed_distribution_fixed_number_iterations(distribution, number_iterations, number_bits_start, number_bits_end, number_trials):
    for number_bits in range(number_bits_start, number_bits_end):
        for algorithm in ["serial_naive", "serial_cache_oblivious", "parallel_naive", "parallel_cache_oblivious"]:
            for trial_num in range(number_trials):
                trial(algorithm, distribution, number_bits, number_iterations)

def serial_fixed_distribution_fixed_number_iterations(distribution, number_iterations, number_bits_start, number_bits_end, number_trials):
    for number_bits in range(number_bits_start, number_bits_end):
        for algorithm in ["serial_naive", "serial_cache_oblivious"]:
            for trial_num in range(number_trials):
                trial(algorithm, distribution, number_bits, number_iterations)

def parallel_fixed_distribution_fixed_number_iterations(distribution, number_iterations, number_bits_start, number_bits_end, number_trials):
    for number_bits in range(number_bits_start, number_bits_end):
        for algorithm in ["parallel_naive", "parallel_cache_oblivious"]:
            for trial_num in range(number_trials):
                trial(algorithm, distribution, number_bits, number_iterations)

def main():
    # trial("serial_naive", "mixed_normal", 27, 50)

    fixed_distribution_fixed_number_iterations("mixed_normal", 50, 28, 30, 5)

if __name__ == "__main__":
    main()