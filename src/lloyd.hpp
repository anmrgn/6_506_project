#pragma once

#include "distribution.hpp"

#include <cilk/cilk.h>
#include <vector>


/**
 * @brief Does one step of Lloyd algorithm, updating representation points in-place
 * 
 * @warning this function assumes that the number of representation points is non-empty.
 * 
 * @param representation_points representation points in increasing order prior to Lloyd step
 * @param distribution The probability distribution being considered
 */
void serial_lloyd_step(std::vector<double> &representation_points, const Distribution &distribution)
{
    double previous_representation_point = Distribution::neg_inf;

    for (size_t idx = 0; idx < representation_points.size() - 1; ++idx)
    {
        // note that previous_representation_point is NOT representation_points[idx - 1] since that just got updated
        double current_representation_point = representation_points[idx];
        double next_representation_point = representation_points[idx + 1];

        double previous_boundary = (previous_representation_point + current_representation_point) / 2.0;
        double next_boundary = (next_representation_point + current_representation_point) / 2.0;

        representation_points[idx] = distribution.conditional_mean(previous_boundary, next_boundary);
        previous_representation_point = current_representation_point;
    }

    double current_representation_point = representation_points[representation_points.size() - 1];
    double previous_boundary = (previous_representation_point + current_representation_point) / 2.0;

    representation_points[representation_points.size() - 1] = distribution.conditional_mean(previous_boundary, Distribution::pos_inf);
}

/**
 * @brief  Run serial Lloyd algorithm for specified number of iterations
 * 
 * @param input_representation_points 
 * @param distribution 
 * @param number_iterations 
 * @return output representation points
 */
std::vector<double> serial_lloyd(const std::vector<double> &input_representation_points, const Distribution &distribution, size_t number_iterations)
{
    std::vector<double> representation_points(input_representation_points);
    for (size_t iteration = 0; iteration < number_iterations; ++iteration)
    {
        serial_lloyd_step(representation_points, distribution);
    }
    return representation_points;
}

/**
 * @brief Does one step of Lloyd algorithm, updating representation points, placing result in separate vector
 * 
 * This runs a single Lloyd step in parallel across representation points. Naive parallel approach.
 * 
 * @param input_representation_points representation points in increasing order prior to Lloyd step
 * @param output_representation_points representation points in increasing order after LLoyd step
 * @param distribution The probability distribution being considered
 */
void parallel_lloyd_step(const std::vector<double> &input_representation_points, std::vector<double> &output_representation_points, const Distribution &distribution)
{
    cilk_for (size_t idx = 0; idx < input_representation_points.size(); ++idx)
    {
        double previous_representation_point = Distribution::neg_inf;
        double current_representation_point = input_representation_points[idx];
        double next_representation_point = Distribution::pos_inf;
        if (idx > 0)
        {
            previous_representation_point = input_representation_points[idx - 1];
        }

        if (idx < input_representation_points.size() - 1)
        {
            next_representation_point = input_representation_points[idx + 1];
        }

        double previous_boundary = (previous_representation_point + current_representation_point) / 2.0;
        double next_boundary = (next_representation_point + current_representation_point) / 2.0;

        output_representation_points[idx] = distribution.conditional_mean(previous_boundary, next_boundary);
    }
}

/**
 * @brief  Run parallel Lloyd algorithm for specified number of iterations
 * 
 * @param input_representation_points 
 * @param distribution 
 * @param number_iterations 
 * @return output representation points
 */
std::vector<double> parallel_lloyd(const std::vector<double> &input_representation_points, const Distribution &distribution, size_t number_iterations)
{
    std::vector<double> current_representation_points(input_representation_points);
    std::vector<double> next_representation_points(current_representation_points.size());

    for (size_t iteration = 0; iteration < number_iterations; ++iteration)
    {
        parallel_lloyd_step(current_representation_points, next_representation_points, distribution);
        std::swap(current_representation_points, next_representation_points);
    }
    return current_representation_points;
}

void lloyd_kernel
(
    size_t t,
    size_t idx,
    const std::vector<double> *input_representation_points,
    std::vector<double> *output_representation_points,
    const Distribution &distribution
)
{
    double previous_representation_point = Distribution::neg_inf;
    double current_representation_point = (*input_representation_points)[idx];
    double next_representation_point = Distribution::pos_inf;
    if (idx > 0)
    {
        previous_representation_point = (*input_representation_points)[idx - 1];
    }

    if (idx < input_representation_points->size() - 1)
    {
        next_representation_point = (*input_representation_points)[idx + 1];
    }

    double previous_boundary = (previous_representation_point + current_representation_point) / 2.0;
    double next_boundary = (next_representation_point + current_representation_point) / 2.0;

    (*output_representation_points)[idx] = distribution.conditional_mean(previous_boundary, next_boundary);
}

/**
 * @brief Implements walk1 algorithm from Frigo and Strumpen's algorithm (2005)
 * 
 * @param t0 
 * @param t1 
 * @param x0 
 * @param dx0_dt 
 * @param x1 
 * @param dx1_dt 
 * @param representation_points_even_iterations 
 * @param representation_points_odd_iterations 
 * @param distribution 
 */
void serial_lloyd_cache_oblivious_walk
(
    int t0,
    int t1,
    int x0,
    int dx0_dt,
    int x1,
    int dx1_dt,
    std::vector<double> *representation_points_even_iterations,
    std::vector<double> *representation_points_odd_iterations,
    const Distribution &distribution
)
{
    int dt = t1 - t0;

    if (dt == 1)
    {
        for (int x = x0; x < x1; ++x)
        {
            const std::vector<double> *input_representation_points = nullptr;
            std::vector<double> *output_representation_points = nullptr;

            if ((t0 % 2) == 0)
            {
                input_representation_points = representation_points_odd_iterations;
                output_representation_points = representation_points_even_iterations;
            }
            else
            {
                input_representation_points = representation_points_even_iterations;
                output_representation_points = representation_points_odd_iterations;
            }

            lloyd_kernel(t0, x, input_representation_points, output_representation_points, distribution);
        }
    }
    else if (dt > 1)
    {
        if (2 * (x1 - x0) + (dx1_dt - dx0_dt) * dt >= 4 * dt)
        {
            int xm = (2 * (x0 + x1) + (2 + dx0_dt + dx1_dt) * dt) / 4;
            serial_lloyd_cache_oblivious_walk(t0, t1, x0, dx0_dt, xm, -1, representation_points_even_iterations, representation_points_odd_iterations, distribution);
            serial_lloyd_cache_oblivious_walk(t0, t1, xm, -1, x1, dx1_dt, representation_points_even_iterations, representation_points_odd_iterations, distribution);
        }
        else
        {
            int s = dt / 2;
            serial_lloyd_cache_oblivious_walk(t0, t0 + s, x0, dx0_dt, x1, dx1_dt, representation_points_even_iterations, representation_points_odd_iterations, distribution);
            serial_lloyd_cache_oblivious_walk(t0 + s, t1, x0 + dx0_dt * s, dx0_dt, x1 + dx1_dt * s, dx1_dt, representation_points_even_iterations, representation_points_odd_iterations, distribution);
        }
    }
    
}

std::vector<double> serial_lloyd_cache_oblivious(const std::vector<double> &input_representation_points, const Distribution &distribution, size_t number_iterations)
{
    const size_t &number_points = input_representation_points.size();
    std::vector<double> representation_points_even_iterations(input_representation_points);
    std::vector<double> representation_points_odd_iterations(number_points);

    // time t = 0 represents initial set of points
    serial_lloyd_cache_oblivious_walk(1, number_iterations + 1, 0, 0, number_points, 0, &representation_points_even_iterations, &representation_points_odd_iterations, distribution);

    if ((number_iterations % 2) == 0)
    {
        return representation_points_even_iterations;
    }
    else
    {
        return representation_points_odd_iterations;
    }
}

/**
 * @brief Implements walk1 algorithm from Frigo and Strumpen's parallel algorithm (2006)
 * 
 * @param t0 
 * @param t1 
 * @param x0 
 * @param dx0_dt 
 * @param x1 
 * @param dx1_dt 
 * @param representation_points_even_iterations 
 * @param representation_points_odd_iterations 
 * @param distribution
 * @param r
 */
void parallel_lloyd_cache_oblivious_walk
(
    int t0,
    int t1,
    int x0,
    int dx0_dt,
    int x1,
    int dx1_dt,
    std::vector<double> *representation_points_even_iterations,
    std::vector<double> *representation_points_odd_iterations,
    const Distribution &distribution,
    int r
)
{
    int h = t1 - t0;
    int dx = x1 - x0;
    int i;

    if ((h >= 1) && (dx >= 2 * h * r))
    {
        int l = dx / r;
        for (i = 0; i < r - 1; ++i)
        {
            cilk_spawn parallel_lloyd_cache_oblivious_walk(
                t0, t1, x0 + i * l, 1, x0 + (i + 1) * l, -1, 
                representation_points_even_iterations, 
                representation_points_odd_iterations, 
                distribution, r
            );
        }
        cilk_spawn parallel_lloyd_cache_oblivious_walk(
            t0, t1, x0 + i * l, 1, x1, -1, 
            representation_points_even_iterations, 
            representation_points_odd_iterations, 
            distribution, r
        );

        cilk_sync;

        cilk_spawn parallel_lloyd_cache_oblivious_walk(
            t0, t1, x0, dx0_dt, x0, 1, 
            representation_points_even_iterations, 
            representation_points_odd_iterations, 
            distribution, r
        );
        for (i = 1; i < r; ++i)
        {
            cilk_spawn parallel_lloyd_cache_oblivious_walk(
                t0, t1, x0 + i * l, -1, x0 + i * l, 1, 
                representation_points_even_iterations, 
                representation_points_odd_iterations, 
                distribution, r
            );
        }
        cilk_spawn parallel_lloyd_cache_oblivious_walk(
            t0, t1, x1, -1, x1, dx1_dt, 
            representation_points_even_iterations, 
            representation_points_odd_iterations, 
            distribution, r
        );
    }
    else if (h > 1)
    {
        int s = h / 2;
        parallel_lloyd_cache_oblivious_walk(
            t0, t0 + s, x0, dx0_dt, x1, dx1_dt, 
            representation_points_even_iterations, 
            representation_points_odd_iterations, 
            distribution, r
        );
        parallel_lloyd_cache_oblivious_walk(
            t0 + s, t1, x0 + dx0_dt * s, dx0_dt, x1 + dx1_dt * s, dx1_dt, 
            representation_points_even_iterations, 
            representation_points_odd_iterations, 
            distribution, r
        );
    }
    else if (h == 1)
    {
        cilk_for (int x = x0; x < x1; ++x)
        {
            const std::vector<double> *input_representation_points = nullptr;
            std::vector<double> *output_representation_points = nullptr;

            if ((t0 % 2) == 0)
            {
                input_representation_points = representation_points_odd_iterations;
                output_representation_points = representation_points_even_iterations;
            }
            else
            {
                input_representation_points = representation_points_even_iterations;
                output_representation_points = representation_points_odd_iterations;
            }

            lloyd_kernel(t0, x, input_representation_points, output_representation_points, distribution);
        }
    }
}

std::vector<double> parallel_lloyd_cache_oblivious(const std::vector<double> &input_representation_points, const Distribution &distribution, size_t number_iterations, int r = 2)
{
    const size_t &number_points = input_representation_points.size();
    std::vector<double> representation_points_even_iterations(input_representation_points);
    std::vector<double> representation_points_odd_iterations(number_points);

    // time t = 0 represents initial set of points
    parallel_lloyd_cache_oblivious_walk(1, number_iterations + 1, 0, 0, number_points, 0, &representation_points_even_iterations, &representation_points_odd_iterations, distribution, r);

    if ((number_iterations % 2) == 0)
    {
        return representation_points_even_iterations;
    }
    else
    {
        return representation_points_odd_iterations;
    }
}

/**
 * @brief Allows for pre-configuration of parameters for the algorithm (e.g., setting r in parallel cache oblivious)
 */

class LloydAlgorithm
{
public:
    virtual ~LloydAlgorithm() = default;
    virtual std::vector<double> run(const std::vector<double> &input_representation_points, const Distribution &distribution, size_t number_iterations) const = 0;
};

class SerialLloydAlgorithm : public LloydAlgorithm
{
public:
    SerialLloydAlgorithm() = default;

    virtual std::vector<double> run(const std::vector<double> &input_representation_points, const Distribution &distribution, size_t number_iterations) const override
    {
        return serial_lloyd(input_representation_points, distribution, number_iterations);
    }
};

class SerialCacheObliviousLloydAlgorithm : public LloydAlgorithm
{
public:
    SerialCacheObliviousLloydAlgorithm() = default;
    
    virtual std::vector<double> run(const std::vector<double> &input_representation_points, const Distribution &distribution, size_t number_iterations) const override
    {
        return serial_lloyd_cache_oblivious(input_representation_points, distribution, number_iterations);
    }
};

class ParallelLloydAlgorithm : public LloydAlgorithm
{
public:
    ParallelLloydAlgorithm() = default;

    virtual std::vector<double> run(const std::vector<double> &input_representation_points, const Distribution &distribution, size_t number_iterations) const override
    {
        return parallel_lloyd(input_representation_points, distribution, number_iterations);
    }
};

class ParallelCacheObliviousLloydAlgorithm : public LloydAlgorithm
{
public:
    ParallelCacheObliviousLloydAlgorithm(int r = 2)
        : m_r(r)
    {
        if (r < 2)
        {
            throw std::invalid_argument("Require r >= 2."); 
        }

    }

    virtual std::vector<double> run(const std::vector<double> &input_representation_points, const Distribution &distribution, size_t number_iterations) const override
    {
        return parallel_lloyd_cache_oblivious(input_representation_points, distribution, number_iterations, m_r);
    }

private:
    const int m_r;
};
