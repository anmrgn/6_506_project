#pragma once

#include <limits>
#include <algorithm>
#include <stdexcept>
#include <cmath>

/**
 * @brief Describes the necessary distribution-specific queries for the Lloyd algorithm
 */
class Distribution
{
public:

    /**
     * @brief Computes the conditional mean in the interval [a, b]
     * 
     * @note This implementation uses the probability integral and mean integral, but if you can simplify the formula, then override this function
     * 
     * @param a lower interval bound
     * @param b upper interval bound
     * @return conditional mean of the random variable inside this interval 
     */
    virtual double conditional_mean(double a, double b) const
    {
        return mean_integral(a, b) / probability_integral(a, b);
    }

    /**
     * @brief Integral of f(x) between a and b, where f(x) is the pdf of the distribution
     * 
     * @warning Require proper handling of the case where a is negative infinity or b is positive infinity
     * 
     * @param a lower integral bound
     * @param b upper integral bound
     * @return result of the integral
     */
    virtual double probability_integral(double a, double b) const = 0;

    /**
     * @brief Integral of x * f(x) between a and b, where f(x) is the pdf of the distribution
     * 
     * @warning Require proper handling of the case where a is negative infinity or b is positive infinity
     * 
     * @param a lower integral bound
     * @param b upper integral bound
     * @return result of the integral
     */
    virtual double mean_integral(double a, double b) const = 0;

    static constexpr double pos_inf = std::numeric_limits<double>::infinity();
    static constexpr double neg_inf = (-1) * std::numeric_limits<double>::infinity();
};

class Uniform : public Distribution
{
public:
    Uniform(double lower_endpoint, double upper_endpoint)
        : m_lower_endpoint(lower_endpoint)
        , m_upper_endpoint(upper_endpoint)
        , m_density(1 / (m_upper_endpoint - m_lower_endpoint))
    {
        if (m_lower_endpoint >= m_upper_endpoint)
        {
            throw std::invalid_argument("Lower bound must not exceed upper bound.");
        }

        if (lower_endpoint == Distribution::neg_inf)
        {
            throw std::invalid_argument("Lower bound must be finite");
        }

        if (upper_endpoint == Distribution::pos_inf)
        {
            throw std::invalid_argument("Upper bound must be finite");
        }
    }

    virtual double conditional_mean(double a, double b) const override
    {
        a = std::max(a, m_lower_endpoint);
        b = std::min(b, m_upper_endpoint);

        return (a + b) / 2;
    }

    virtual double probability_integral(double a, double b) const override
    {
        a = std::max(a, m_lower_endpoint);
        b = std::min(b, m_upper_endpoint);

        return m_density * (b - a);
    }

    virtual double mean_integral(double a, double b) const override
    {
        a = std::max(a, m_lower_endpoint);
        b = std::min(b, m_upper_endpoint);

        return m_density * (b * b - a * a) / 2;
    }

    const double m_lower_endpoint;
    const double m_upper_endpoint;
    const double m_density;
};

class Normal : public Distribution
{
public:

    Normal(double mean, double var)
        : m_mean(mean)
        , m_std(std::sqrt(var))
    {
        if ((m_mean == Distribution::pos_inf) || (m_mean == Distribution::neg_inf))
        {
            throw std::invalid_argument("Mean must be finite");
        }

        if ((var <= 0) || (var == Distribution::pos_inf))
        {
            throw std::invalid_argument("Variance must be positive and finite");
        }
    }

    virtual double probability_integral(double a, double b) const override
    {
        double cdf_a = std_cdf((a - m_mean) / m_std);
        double cdf_b = std_cdf((b - m_mean) / m_std);

        return cdf_b - cdf_a;
    }

    virtual double mean_integral(double a, double b) const override
    {
        constexpr double factor = M_2_SQRTPI * M_SQRT1_2 / 2.0;

        double norm_a = (a - m_mean) / (M_SQRT2 * m_std);
        double indef_int_a = m_mean * std::erf(norm_a) / 2.0
                - m_std * std::exp(-norm_a * norm_a) * factor;

        double norm_b = (b - m_mean) / (M_SQRT2 * m_std);
        double indef_int_b = m_mean * std::erf(norm_b) / 2.0
                - m_std * std::exp(-norm_b * norm_b) * factor;

        return indef_int_b - indef_int_a;
    }

    inline double std_cdf(double a) const
    {
        return (1.0 + std::erf(a / M_SQRT2)) / 2.0;
    }

    const double m_mean;
    const double m_std;
};