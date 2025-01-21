#pragma once

#include <vector>

bool serial_conj_grad(const std::vector<float>& in_b_vec, std::vector<float>& out_x_vec, const float threshold, const int max_iters);
