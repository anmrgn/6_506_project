#include "distribution.hpp"
#include "lloyd.hpp"

#include <memory>
#include <algorithm>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <limits>
#include <chrono>
#include <sstream>

#define ERROR -1

constexpr unsigned int NS_PER_MS = 1000000;
constexpr double EPSILON = 1e-9;

void print_representation_points(const std::vector<double> &representation_points)
{
    for (size_t idx = 0; idx < representation_points.size(); ++idx)
    {
        std::cout << representation_points[idx] << " ";
    }
    std::cout << std::endl;
}

std::vector<double> linspace(double start, double end, size_t num)
{
    if (num < 2)
    {
        throw std::invalid_argument("Require num >= 2 because I don't want to deal with edge cases");
    }

    double step = (end - start) / ((double) (num - 1));
    std::vector<double> result(num);

    double curr = start;
    for (size_t idx = 0; idx < result.size(); ++idx)
    {
        result[idx] = curr;
        curr += step;
    }
    return result;
}

bool all_equal(const std::vector<double> &array_1, const std::vector<double> &array_2)
{
    if (array_1.size() != array_2.size())
    {
        return false;
    }
    
    for (size_t idx = 0; idx < array_1.size(); ++idx)
    {
        if (std::abs(array_1[idx] - array_2[idx]) > EPSILON)
        {
            return false;
        }
    }
    return true;
}

void manual_test()
{
    Uniform std_uniform(0.0, 1.0);
    Normal std_normal(0.0, 1.0);

    std::shared_ptr<Normal> normal_left = std::make_shared<Normal>(-5.0, 1.0);
    std::shared_ptr<Normal> normal_right = std::make_shared<Normal>(5.0, 1.0);
    MixtureDistribution mixed_normal({normal_left, normal_right}, {0.5, 0.5});

    // 8 representation points means 3 bits per sample
    // std::vector<double> input_representation_points = {-1.1, -0.9, -0.7, -0.5, 0.5, 0.7, 0.9, 1.1};
    size_t number_bits = 20;
    size_t number_representation_points = (1 << number_bits);
    std::vector<double> input_representation_points = linspace(-5, 5, number_representation_points);
    // print_representation_points(input_representation_points);

    size_t number_iterations = 50;

    const std::chrono::steady_clock::time_point serial_start = std::chrono::steady_clock::now();
    std::vector<double> serial_result = serial_lloyd(input_representation_points, mixed_normal, number_iterations);
    const std::chrono::steady_clock::time_point serial_end = std::chrono::steady_clock::now();
    
    const int64_t serial_duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(serial_end - serial_start).count();
    
    const std::chrono::steady_clock::time_point serial_cache_oblivious_start = std::chrono::steady_clock::now();
    std::vector<double> serial_cache_oblivious_result = serial_lloyd_cache_oblivious(input_representation_points, mixed_normal, number_iterations);
    const std::chrono::steady_clock::time_point serial_cache_oblivious_end = std::chrono::steady_clock::now();
    
    const int64_t serial_cache_oblivious_duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(serial_cache_oblivious_end - serial_cache_oblivious_start).count();

    const std::chrono::steady_clock::time_point parallel_start = std::chrono::steady_clock::now();
    std::vector<double> parallel_result = parallel_lloyd(input_representation_points, mixed_normal, number_iterations);
    const std::chrono::steady_clock::time_point parallel_end = std::chrono::steady_clock::now();
    
    const int64_t parallel_duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(parallel_end - parallel_start).count();

    const std::chrono::steady_clock::time_point parallel_cache_oblivious_start = std::chrono::steady_clock::now();
    std::vector<double> parallel_cache_oblivious_result = parallel_lloyd_cache_oblivious(input_representation_points, mixed_normal, number_iterations);
    const std::chrono::steady_clock::time_point parallel_cache_oblivious_end = std::chrono::steady_clock::now();
    
    const int64_t parallel_cache_oblivious_duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(parallel_cache_oblivious_end - parallel_cache_oblivious_start).count();

    // print_representation_points(serial_result);
    // print_representation_points(serial_cache_oblivious_result);
    // print_representation_points(parallel_result);
    // print_representation_points(parallel_cache_oblivious_result);

    std::cout << "Serial time (ms): " << serial_duration_ns / NS_PER_MS << std::endl;
    std::cout << "Serial cache oblivious time (ms): " << serial_cache_oblivious_duration_ns / NS_PER_MS << std::endl;
    std::cout << "Parallel time (ms): " << parallel_duration_ns / NS_PER_MS << std::endl;
    std::cout << "Parallel cache oblivious time (ms): " << parallel_cache_oblivious_duration_ns / NS_PER_MS << std::endl;

    std::cout << "Serial matches serial cache oblivious: " << all_equal(serial_result, serial_cache_oblivious_result) << std::endl;
    std::cout << "Serial matches parallel: " << all_equal(serial_result, parallel_result) << std::endl;
    std::cout << "Serial matches parallel cache oblivious: " << all_equal(serial_result, parallel_cache_oblivious_result) << std::endl;
}

struct RunSpecification
{
public:

    RunSpecification(std::unique_ptr<const LloydAlgorithm> &&algorithm, const std::vector<double> &input_representation_points, std::unique_ptr<const Distribution> &&distribution, size_t number_iterations)
        : algorithm(std::move(algorithm))
        , input_representation_points(input_representation_points)
        , distribution(std::move(distribution))
        , number_iterations(number_iterations)
    {

    }

    std::vector<double> run()
    {
        return algorithm->run(input_representation_points, *distribution, number_iterations);
    }

    const std::unique_ptr<const LloydAlgorithm> algorithm;
    const std::vector<double> input_representation_points;
    const std::unique_ptr<const Distribution> distribution;
    const size_t number_iterations;

};

std::unique_ptr<LloydAlgorithm> string_to_algorithm(const std::string &algorithm_string)
{
    if (algorithm_string == "serial_naive")
    {
        return std::make_unique<SerialLloydAlgorithm>();
    }
    else if (algorithm_string == "serial_cache_oblivious")
    {
        return std::make_unique<SerialCacheObliviousLloydAlgorithm>();
    }
    else if (algorithm_string == "parallel_naive")
    {
        return std::make_unique<ParallelLloydAlgorithm>();
    }
    else if (algorithm_string == "parallel_cache_oblivious")
    {
        return std::make_unique<ParallelCacheObliviousLloydAlgorithm>();
    }
    else
    {
        throw std::invalid_argument("Unknown algorithm.");
    }
}

std::unique_ptr<Distribution> string_to_distribution(const std::string &distribution_string)
{
    if (distribution_string == "standard_normal")
    {
        return std::make_unique<Normal>(0.0, 1.0);
    }
    else if (distribution_string == "uniform")
    {
        return std::make_unique<Uniform>(-2.0, 2.0);
    }
    else if (distribution_string == "mixed_normal")
    {
        std::shared_ptr<Normal> normal_left = std::make_shared<Normal>(-2.0, 1.0);
        std::shared_ptr<Normal> normal_right = std::make_shared<Normal>(2.0, 1.0);
        std::vector<std::shared_ptr<const Distribution>> distributions{normal_left, normal_right};
        
        std::vector<double> weights{0.5, 0.5};
        
        return std::make_unique<MixtureDistribution>(distributions, weights);
    }
    else if (distribution_string == "mixed_uniform")
    {
        std::shared_ptr<Uniform> uniform_left = std::make_shared<Uniform>(-2.0, 1.0);
        std::shared_ptr<Uniform> uniform_right = std::make_shared<Uniform>(-1.0, 2.0);
        std::vector<std::shared_ptr<const Distribution>> distributions{uniform_left, uniform_right};

        std::vector<double> weights{0.5, 0.5};

        return std::make_unique<MixtureDistribution>(distributions, weights);
    }
    else
    {
        throw std::invalid_argument("Unknown distribution.");
    }
}

size_t string_to_number_bits(const std::string &number_bits_string)
{
    std::stringstream ss(number_bits_string);
    size_t number_bits;
    ss >> number_bits;
    return number_bits;
}

size_t string_to_number_iterations(const std::string &number_iterations_string)
{
    std::stringstream ss(number_iterations_string);
    size_t number_iterations;
    ss >> number_iterations;
    return number_iterations;
}


RunSpecification parse_args(int argc, char **argv)
{
    // format: ./bin/main alg dist n_bits n_iter

    if (argc != 5)
    {
        throw std::invalid_argument("Command line arguments must be of the form ./bin/main algorithm distribution number_bits number_iterations.");
    }

    std::string algorithm_string(argv[1]);
    std::string distribution_string(argv[2]);
    std::string number_bits_string(argv[3]);
    std::string number_iterations_string(argv[4]);

    std::unique_ptr<LloydAlgorithm> algorithm = string_to_algorithm(algorithm_string);
    std::unique_ptr<Distribution> distribution = string_to_distribution(distribution_string);
    size_t number_bits = string_to_number_bits(number_bits_string);
    size_t number_iterations = string_to_number_iterations(number_iterations_string);
    size_t number_representation_points = (1 << number_bits);
    std::vector<double> input_representation_points = linspace(-1.0, 1.0, number_representation_points);

    return RunSpecification(std::move(algorithm), input_representation_points, std::move(distribution), number_iterations);
}

int main(int argc, char **argv)
{
    if (!std::numeric_limits<double>::has_infinity)
    {
        std::cerr << "Require infinity as possible input for the distribution class to work." << std::endl;
        return ERROR;
    }

    // manual_test();

    try
    {
        RunSpecification run_specification = parse_args(argc, argv);
        const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
        run_specification.run();
        const std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        const int64_t duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        std::cout << duration_ns << std::endl;
    }
    catch (const std::invalid_argument &error)
    {
        std::cerr << error.what() << std::endl;
        return ERROR;
    }
    
    return 0;
}