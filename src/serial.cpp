#include "serial.h"
#include "utils.h"

using std::vector;

// Matrix-by-vector multiply by using a specific sparse matrix of 3 diagonals:
// | -2  1  0  0 ...  0  0  0 |
// |  1 -2  1  0 ...  0  0  0 |
// |  0  1 -2  1 ...  0  0  0 |
// |  0  0  1 -2 ...  0  0  0 |
//      ...      ...    ...  
// |  0  0  0  0 ... -2  1  0 |     
// |  0  0  0  0 ...  1 -2  1 |
// |  0  0  0  0 ...  0  1 -2 |
void serial_sparse_matmul(vector<float>& dest, const vector<float>& in_x_vec)
{
    dest.resize(in_x_vec.size());
    const size_t N = in_x_vec.size();

    // Handle edge cases first
    dest[0] = in_x_vec[0] * -2.0f + in_x_vec[1];
    dest[N-1] = in_x_vec[N-1] * -2.0f + in_x_vec[N-2];

    for (size_t i = 1; i < N-1; i++)
    {
        float result = in_x_vec[i] * -2.0f;
        result += in_x_vec[i-1];
        result += in_x_vec[i+1];
        dest[i] = result;
    }
}

bool serial_conj_grad(const vector<float>& in_b_vec, vector<float>& out_x_vec, const float threshold, const int max_iters)
{
    vector<float> x_vec = vector<float>(in_b_vec.size(), 0.0f);

    // NOTE: Most of this is redundant for x_vec initialized to zero
    vector<float> temp;
    serial_sparse_matmul(temp, x_vec);
    vector<float> r_vec = in_b_vec;
    vector_sub(r_vec, temp);
    vector<float> p_vec = r_vec;

    float r_dot_r = vector_dot(r_vec, r_vec);

    for (int k = 0; k < max_iters; k++)
    {
        serial_sparse_matmul(temp, p_vec);
        const float p_dot_a_p = vector_dot(p_vec, temp);
        const float alpha = r_dot_r / p_dot_a_p;

        vector_axby(x_vec, x_vec, 1.0, p_vec, alpha);
        vector_axby(r_vec, r_vec, 1.0, temp, -alpha);

        const float r_abs_max = vector_absmax(r_vec);
        if (r_abs_max <= threshold)
        {
            // We've reached an approximate solution within threshold
            out_x_vec = x_vec;
            return true;
        }

        const float new_r_dot_r = vector_dot(r_vec, r_vec);
        const float beta = new_r_dot_r / r_dot_r;
        r_dot_r = new_r_dot_r;

        vector_axby(p_vec, r_vec, 1.0, p_vec, beta);
    }

    return false; // Exceeded max iterations
}