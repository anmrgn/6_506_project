#include "distribution.hpp"
#include "serial_lloyd.hpp"

#include <algorithm>
#include <vector>
#include <iostream>
#include <limits>

void print_representation_points(const std::vector<double> &representation_points)
{
    for (size_t idx = 0; idx < representation_points.size(); ++idx)
    {
        std::cout << representation_points[idx] << " ";
    }
    std::cout << std::endl;
}
int main(int argc, char **argv)
{
    if (!std::numeric_limits<double>::has_infinity)
    {
        std::cerr << "Require infinity as possible input for the distribution class to work." << std::endl;
        return -1;
    }

    Uniform uniform(0, 1);
    Normal normal(0, 1);

    std::vector<double> representation_points = {0.1, 0.4, 0.9};

    for (unsigned int iteration = 0; iteration < 100; ++iteration)
    {
        print_representation_points(representation_points);
        serial_lloyd_step(representation_points, normal);
    }
    
    return 0;
}