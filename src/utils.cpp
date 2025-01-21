#include "utils.h"

#include <utility>
#include <string>
#include <iomanip>
#include <chrono>

using std::vector;


void vector_add(vector<float>& dest, const vector<float>& src)
{
    for (size_t i = 0; i < std::min(dest.size(), src.size()); i++)
    {
        dest[i] += src[i];
    }
}

void vector_axby(std::vector<float>& dest, const std::vector<float>& a, float x, const std::vector<float>& b, float y)
{
    size_t N = std::min(a.size(), b.size());
    dest.resize(N);
    for (size_t i = 0; i < N; i++)
    {
        dest[i] = a[i] * x + b[i] * y;
    }
}


void vector_sub(vector<float>& dest, const vector<float>& src)
{
    for (size_t i = 0; i < std::min(dest.size(), src.size()); i++)
    {
        dest[i] -= src[i];
    }
}

float vector_dot(const vector<float>& a, const vector<float>& b)
{
    float result = 0;
    for (size_t i = 0; i < std::min(a.size(), b.size()); i++)
    {
        result += a[i] * b[i];
    }
    return result;
}

float vector_absmax(const std::vector<float>& a)
{
    float result = 0;
    for (float f : a)
    {
        result = std::max(result, std::abs(f));
    }
    return result;
}


void read_from_file(std::istream& file, vector<float>& dest)
{
    dest.clear();
    std::string line;

    while (std::getline(file, line))
    {
        float f = std::stof(line);
        dest.push_back(f);
    }
}

void write_to_file(std::ostream& file, const std::vector<float>& data)
{
    file << std::scientific << std::setprecision(10);
    for (float f : data)
    {
        file << f << std::endl;
    }
}

int64_t timestamp_ns()
{
    auto now = std::chrono::high_resolution_clock::now();
    auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch());
    return now_ns.count();
}