#include "distribution.hpp"
#include "lloyd.hpp"

#include <memory>
#include <algorithm>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <limits>
#include <chrono>

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

int main(int argc, char **argv)
{
    if (!std::numeric_limits<double>::has_infinity)
    {
        std::cerr << "Require infinity as possible input for the distribution class to work." << std::endl;
        return -1;
    }

    Uniform std_uniform(0.0, 1.0);
    Normal std_normal(0.0, 1.0);

    std::shared_ptr<Normal> normal_left = std::make_shared<Normal>(-5.0, 1.0);
    std::shared_ptr<Normal> normal_right = std::make_shared<Normal>(5.0, 1.0);
    MixtureDistribution mixed_normal({normal_left, normal_right}, {0.5, 0.5});

    // 8 representation points means 3 bits per sample
    // std::vector<double> input_representation_points = {-1.1, -0.9, -0.7, -0.5, 0.5, 0.7, 0.9, 1.1};
    size_t number_bits = 16;
    std::vector<double> input_representation_points = linspace(-5, 5, (1 << number_bits));
    // print_representation_points(input_representation_points);

    size_t number_iterations = 100;

    const std::chrono::steady_clock::time_point serial_start = std::chrono::steady_clock::now();
    std::vector<double> serial_result = serial_lloyd(input_representation_points, mixed_normal, number_iterations);
    const std::chrono::steady_clock::time_point serial_end = std::chrono::steady_clock::now();
    
    const int64_t serial_duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(serial_end - serial_start).count();
    
    const std::chrono::steady_clock::time_point parallel_start = std::chrono::steady_clock::now();
    std::vector<double> parallel_result = parallel_lloyd(input_representation_points, mixed_normal, number_iterations);
    const std::chrono::steady_clock::time_point parallel_end = std::chrono::steady_clock::now();
    
    const int64_t parallel_duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(parallel_end - parallel_start).count();

    // print_representation_points(serial_result);
    // print_representation_points(parallel_result);

    std::cout << "Serial time (ms): " << serial_duration_ns / 1000000 << std::endl;
    std::cout << "Parallel time (ms): " << parallel_duration_ns / 1000000 << std::endl;

    return 0;
}