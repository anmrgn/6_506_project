#include "distribution.hpp"

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