#pragma once

#include <fstream>
#include <vector>

constexpr int div_round_up(int x, int y)
{
    return (x + y - 1) / y;
}

void vector_add(std::vector<float>& dest, const std::vector<float>& src);
void vector_axby(std::vector<float>& dest, const std::vector<float>& a, float x, const std::vector<float>& b, float y);

void vector_sub(std::vector<float>& dest, const std::vector<float>& src);
float vector_dot(const std::vector<float>& a, const std::vector<float>& b);
float vector_absmax(const std::vector<float>& a);

void read_from_file(std::istream& file, std::vector<float>& result);
void write_to_file(std::ostream& file, const std::vector<float>& data);

int64_t timestamp_ns();
