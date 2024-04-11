#include "distribution.hpp"
#include "lloyd.hpp"

#include <memory>
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

    Uniform std_uniform(0.0, 1.0);
    Normal std_normal(0.0, 1.0);

    std::shared_ptr<Normal> normal_left = std::make_shared<Normal>(-5.0, 1.0);
    std::shared_ptr<Normal> normal_right = std::make_shared<Normal>(5.0, 1.0);
    MixtureDistribution mixed_normal({normal_left, normal_right}, {0.5, 0.5});

    // 8 representation points means 3 bits per sample
    std::vector<double> representation_points = {-1.1, -0.9, -0.7, -0.5, 0.5, 0.7, 0.9, 1.1};

    for (unsigned int iteration = 0; iteration < 1000; ++iteration)
    {
        print_representation_points(representation_points);
        serial_lloyd_step(representation_points, mixed_normal);
    }
    
    return 0;
}