#include "parallel.h"
#include "utils.h"

#include <iostream>
#include <cassert>

using std::vector;

__constant__ int d_vec_size;

constexpr int BLOCK_SIZE = 128;

// CUDA kernel that calculates a matrix-by-vector product
// where A is the sparse matrix A of 3 diagonals (1, -2, 1)
__global__ void parallel_sp_matmul(float* dest, const float* x_vec)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;

    // Hoping that these will be converted to conditional/masked loads
    const float left = (i < 1) ? 0 : x_vec[i-1];
    const float right = (i < d_vec_size) ? x_vec[i+1] : 0;

    if (i < d_vec_size)
    {
        float result = x_vec[i] * -2;
        result += left + right;
        dest[i] = result;
    }
}

#define WARP_MASK 0xFFFFFFFF

// CUDA kernel that calculates a vector dot product
// NOTE: Destination *must* be zeroed prior to calling
__global__ void parallel_dot(float* dest, const float* a_vec, const float* b_vec)
{   
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    const int lane_idx = threadIdx.x & 0x1F;

    if (i >= d_vec_size)
    {
        return;
    }

    float accum = a_vec[i] * b_vec[i];
    
    // Warp-level add-reduce
    accum += __shfl_down_sync(WARP_MASK, accum, 16);
    accum += __shfl_down_sync(WARP_MASK, accum, 8);
    accum += __shfl_down_sync(WARP_MASK, accum, 4);
    accum += __shfl_down_sync(WARP_MASK, accum, 2);
    accum += __shfl_down_sync(WARP_MASK, accum, 1);

    // Accumulate into the global destination buffer, this could be a little more optimal
    // if we reduce into a block shared buffer first, before the global atomic add
    if (lane_idx == 0)
    {
        atomicAdd(dest, accum);
    }
}

// CUDA kernel that calculates the maximum (absolute) value for the whole vector
// NOTE: Destination *must* be zeroed prior to calling
__global__ void parallel_absmax(float* dest, const float* a_vec)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    const int lane_idx = threadIdx.x & 0x1F;

    if (i >= d_vec_size)
    {
        return;
    }

    float accum = abs(a_vec[i]);
    
    // Warp-level add-reduce
    accum = max(accum, __shfl_down_sync(WARP_MASK, accum, 16));
    accum = max(accum, __shfl_down_sync(WARP_MASK, accum, 8));
    accum = max(accum, __shfl_down_sync(WARP_MASK, accum, 4));
    accum = max(accum, __shfl_down_sync(WARP_MASK, accum, 2));
    accum = max(accum, __shfl_down_sync(WARP_MASK, accum, 1));

    // Accumulate into the global destination buffer, this could be a little more optimal
    // if we reduce into a block shared buffer first, before the global atomic max
    if (lane_idx == 0)
    {
        // This *should* be ok since we can assume that the sign bit will always be zero
        int a = __float_as_int(accum);
        atomicMax(reinterpret_cast<int*>(dest), a);
    }
}

// CUDA kernel that calculates a scaled element-wise vector sum
__global__ void parallel_axby(float* dest, const float* a_vec, const float* x, const float* b_vec, const float* y)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < d_vec_size)
    {
        const float result = a_vec[i] * (*x) + b_vec[i] * (*y);
        dest[i] = result;
    }
}

// CUDA kernel that calculates a single-thread scalar division, with result and negated result.
// Used for calculating the `alpha` and `beta` scalar values on device
__global__ void scalar_div(float* dest, float* neg_dest, const float* x, const float* y)
{
    float result = *x / *y;
    *dest = result;
    *neg_dest = -result;
}

cudaError_t verify(cudaError_t result)
{
#ifndef NDEBUG
    if (result != cudaSuccess)
    {
        std::cout << "CUDA error: " << cudaGetErrorString(result) << std::endl;
        assert(!"CUDA call failed");
    }
#endif

    return result;   
}

#define LAUNCH_DOMAIN   div_round_up(size, BLOCK_SIZE), BLOCK_SIZE

bool parallel_conj_grad(const vector<float>& in_b_vec, vector<float>& out_x_vec, const float threshold, const int max_iters)
{
    if (in_b_vec.size() >= INT_MAX)
    {
        return false;
    }
    int size = static_cast<int>(in_b_vec.size());
    verify(cudaMemcpyToSymbol(d_vec_size, &size, sizeof(d_vec_size)));

    float* d_x_vec;
    float* d_r_vec;
    float* d_p_vec;
    float* d_a_mul_p;
    float* d_result;
    verify(cudaMalloc(&d_x_vec, size * sizeof(float)));
    verify(cudaMalloc(&d_r_vec, size * sizeof(float)));
    verify(cudaMalloc(&d_p_vec, size * sizeof(float)));
    verify(cudaMalloc(&d_a_mul_p, size * sizeof(float)));
    verify(cudaMalloc(&d_result, size * sizeof(float)));

    float* d_one;
    float* d_neg_one;
    float* d_alpha;
    float* d_neg_alpha;
    float* d_beta;
    float* d_rdotr;
    float* d_new_rdotr;
    float* d_p_dot_a_p;
    float* d_abs_max;
    verify(cudaMalloc(&d_one, sizeof(float)));
    verify(cudaMalloc(&d_neg_one, sizeof(float)));
    verify(cudaMalloc(&d_alpha, sizeof(float)));
    verify(cudaMalloc(&d_neg_alpha, sizeof(float)));
    verify(cudaMalloc(&d_beta, sizeof(float)));
    verify(cudaMalloc(&d_rdotr, sizeof(float)));
    verify(cudaMalloc(&d_new_rdotr, sizeof(float)));
    verify(cudaMalloc(&d_p_dot_a_p, sizeof(float)));
    verify(cudaMalloc(&d_abs_max, sizeof(float)));

    const float one = 1.0;
    const float neg_one = -1.0;

    verify(cudaMemset(d_x_vec, 0, size * sizeof(float)));
    verify(cudaMemset(d_a_mul_p, 0, size * sizeof(float)));
    verify(cudaMemset(d_rdotr, 0, sizeof(float)));

    verify(cudaMemcpy(d_alpha, &one, sizeof(float), cudaMemcpyHostToDevice));
    verify(cudaMemcpy(d_one, &one, sizeof(float), cudaMemcpyHostToDevice));
    verify(cudaMemcpy(d_neg_one, &neg_one, sizeof(float), cudaMemcpyHostToDevice));
    verify(cudaMemcpy(d_r_vec, in_b_vec.data(), size * sizeof(float), cudaMemcpyHostToDevice));
    verify(cudaMemcpy(d_p_vec, d_r_vec, size * sizeof(float), cudaMemcpyDeviceToDevice));

    // parallel_sp_matmul<<<LAUNCH_DOMAIN>>>(d_a_mul_p, d_x_vec);
    // parallel_axby<<<LAUNCH_DOMAIN>>>(d_r_vec, d_b_vec, d_one, d_a_mul_p, d_neg_one);
    parallel_dot<<<LAUNCH_DOMAIN>>>(d_rdotr, d_r_vec, d_r_vec);

    for (int k = 0; k < max_iters; k++)
    {
        parallel_sp_matmul<<<LAUNCH_DOMAIN>>>(d_a_mul_p, d_p_vec);

        verify(cudaMemset(d_p_dot_a_p, 0, sizeof(float)));
        parallel_dot<<<LAUNCH_DOMAIN>>>(d_p_dot_a_p, d_p_vec, d_a_mul_p);
        scalar_div<<<1, 1>>>(d_alpha, d_neg_alpha, d_rdotr, d_p_dot_a_p);

        parallel_axby<<<LAUNCH_DOMAIN>>>(d_x_vec, d_x_vec, d_one, d_p_vec, d_alpha);
        parallel_axby<<<LAUNCH_DOMAIN>>>(d_r_vec, d_r_vec, d_one, d_a_mul_p, d_neg_alpha);

        verify(cudaMemset(d_abs_max, 0, sizeof(float)));
        parallel_absmax<<<LAUNCH_DOMAIN>>>(d_abs_max, d_r_vec);

        float abs_max = 0;
        verify(cudaMemcpy(&abs_max, d_abs_max, sizeof(abs_max), cudaMemcpyDeviceToHost));
        if (abs_max <= threshold)
        {
            // We've reached an approximate solution within threshold
            out_x_vec.resize(size);
            verify(cudaMemcpy(out_x_vec.data(), d_x_vec, size * sizeof(float), cudaMemcpyDeviceToHost));
            return true;
        }

        verify(cudaMemset(d_new_rdotr, 0, sizeof(float)));
        parallel_dot<<<LAUNCH_DOMAIN>>>(d_new_rdotr, d_r_vec, d_r_vec);

        // NOTE: `d_neg_alpha` is used as a dummy destination here
        scalar_div<<<1, 1>>>(d_beta, d_neg_alpha, d_new_rdotr, d_rdotr);
        verify(cudaMemcpy(d_rdotr, d_new_rdotr, sizeof(float), cudaMemcpyDeviceToDevice));

        parallel_axby<<<LAUNCH_DOMAIN>>>(d_p_vec, d_r_vec, d_one, d_p_vec, d_beta);
    }

    vector<float> result(in_b_vec.size(), 0.0f);
    verify(cudaMemcpy(result.data(), d_result, size * sizeof(float), cudaMemcpyDeviceToHost));

    // TODO:
    return false;
}



