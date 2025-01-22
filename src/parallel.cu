#include "parallel.h"
#include "utils.h"

#include <iostream>
#include <cassert>

using std::vector;

__constant__ int d_vec_size;

constexpr int BLOCK_SIZE = 256;


// CUDA kernel helper since there's no native float AtomicMax
__device__ void atomicAbsMaxFloat(float* dest, float value)
{
    // This *should* be safe when the sign bit is always zero
    atomicMax(reinterpret_cast<int*>(dest), __float_as_int(abs(value)));
}


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
        atomicAbsMaxFloat(dest, accum);
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

__global__ void fused_matmul_dot(float *matmul_dest, float* dot_dest, const float* p_vec)
{
    const int global_i = blockDim.x * blockIdx.x + threadIdx.x;
    const int lane_i = threadIdx.x & 0x1F;
    const int i = threadIdx.x + 1;

    __shared__ float p_cache[BLOCK_SIZE + 2];
    __shared__ float dot_accum;

    // Init cache for source data
    p_cache[i] = (global_i < d_vec_size) ? p_vec[global_i] : 0;
    // Init boundary values
    if (threadIdx.x == 0)
    {
        dot_accum = 0;
        p_cache[0] = (global_i < 1) ? 0 : p_vec[global_i-1];
        p_cache[BLOCK_SIZE+1] = ((global_i+BLOCK_SIZE) < d_vec_size) ? p_vec[global_i+BLOCK_SIZE] : 0;
    }

    __syncthreads();
    
    float result = 0;
    if (global_i < d_vec_size)
    {
        result += p_cache[i] * -2;
        result += p_cache[i-1];
        result += p_cache[i+1];
        matmul_dest[global_i] = result;
    }

    // Fused wave-reduce dot product with p_vec
    result *= p_cache[i];
    result += __shfl_down_sync(WARP_MASK, result, 16);
    result += __shfl_down_sync(WARP_MASK, result, 8);
    result += __shfl_down_sync(WARP_MASK, result, 4);
    result += __shfl_down_sync(WARP_MASK, result, 2);
    result += __shfl_down_sync(WARP_MASK, result, 1);
    
    if (lane_i == 0)
    {
        atomicAdd(&dot_accum, result);
    }
    
    __syncthreads();

    if (threadIdx.x == 0)
    {
        atomicAdd(dot_dest, dot_accum);
    }
}

__global__ void fused_vecmul_absmax_dot(float *dest_max, float* dest_dot, const float* rdotr, const float* pdotap,
                                        float* x_vec, float* r_vec, const float* p_vec, const float* a_mul_p)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    const int lane_i = threadIdx.x & 0x1F;
    const float alpha = *rdotr / *pdotap;

    __shared__ float dot_accum;
    __shared__ float max_accum;

    if (threadIdx.x == 0)
    {
        dot_accum = 0;
        max_accum = 0;
    }

    __syncthreads();

    float r = 0;
    if (i < d_vec_size)
    {   
        const float x = x_vec[i] + p_vec[i] * alpha; 
        x_vec[i] = x;

        r = r_vec[i] - a_mul_p[i] * alpha;
        r_vec[i] = r;
    }

    float r_max = abs(r);
    float r_dot = r * r;
    
    r_max = max(r_max, __shfl_down_sync(WARP_MASK, r_max, 16));
    r_dot += __shfl_down_sync(WARP_MASK, r_dot, 16);

    r_max = max(r_max, __shfl_down_sync(WARP_MASK, r_max, 8));
    r_dot += __shfl_down_sync(WARP_MASK, r_dot, 8);

    r_max = max(r_max, __shfl_down_sync(WARP_MASK, r_max, 4));
    r_dot += __shfl_down_sync(WARP_MASK, r_dot, 4);

    r_max = max(r_max, __shfl_down_sync(WARP_MASK, r_max, 2));
    r_dot += __shfl_down_sync(WARP_MASK, r_dot, 2);

    r_max = max(r_max, __shfl_down_sync(WARP_MASK, r_max, 1));
    r_dot += __shfl_down_sync(WARP_MASK, r_dot, 1);

    if (lane_i == 0)
    {
        atomicAdd(&dot_accum, r_dot);
        atomicAbsMaxFloat(&max_accum, r_max);
    }

    __syncthreads();

    if (threadIdx.x == 0)
    {
        atomicAdd(dest_dot, dot_accum);
        atomicAbsMaxFloat(dest_max, max_accum);
    }
}

__global__ void fused_beta_finalize(const float* rdotr, const float* new_rdotr, float *p_vec, const float* r_vec)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    const float beta = *new_rdotr / *rdotr;

    if (i < d_vec_size)
    {
        p_vec[i] = r_vec[i] + p_vec[i] * beta;
    }
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

#define USE_FUSED_KERNELS 1
#define LAUNCH_DOMAIN     div_round_up(size, BLOCK_SIZE), BLOCK_SIZE

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

    parallel_dot<<<LAUNCH_DOMAIN>>>(d_rdotr, d_r_vec, d_r_vec);

    for (int k = 0; k < max_iters; k++)
    {
        verify(cudaMemset(d_p_dot_a_p, 0, sizeof(float)));
        verify(cudaMemset(d_abs_max, 0, sizeof(float)));
        verify(cudaMemset(d_new_rdotr, 0, sizeof(float)));

#if USE_FUSED_KERNELS
        fused_matmul_dot<<<LAUNCH_DOMAIN>>>(d_a_mul_p, d_p_dot_a_p, d_p_vec);
        fused_vecmul_absmax_dot<<<LAUNCH_DOMAIN>>>(d_abs_max, d_new_rdotr, d_rdotr, d_p_dot_a_p, d_x_vec, d_r_vec, d_p_vec, d_a_mul_p);
        fused_beta_finalize<<<LAUNCH_DOMAIN>>>(d_rdotr, d_new_rdotr, d_p_vec, d_r_vec);
        verify(cudaMemcpyAsync(d_rdotr, d_new_rdotr, sizeof(float), cudaMemcpyDeviceToDevice));
#else
        parallel_sp_matmul<<<LAUNCH_DOMAIN>>>(d_a_mul_p, d_p_vec); 

        parallel_dot<<<LAUNCH_DOMAIN>>>(d_p_dot_a_p, d_p_vec, d_a_mul_p);
        scalar_div<<<1, 1>>>(d_alpha, d_neg_alpha, d_rdotr, d_p_dot_a_p);

        parallel_axby<<<LAUNCH_DOMAIN>>>(d_x_vec, d_x_vec, d_one, d_p_vec, d_alpha);
        parallel_axby<<<LAUNCH_DOMAIN>>>(d_r_vec, d_r_vec, d_one, d_a_mul_p, d_neg_alpha);

        parallel_absmax<<<LAUNCH_DOMAIN>>>(d_abs_max, d_r_vec);
#endif // USE_FUSED_KERNELS

        float abs_max = 0;
        verify(cudaMemcpy(&abs_max, d_abs_max, sizeof(abs_max), cudaMemcpyDeviceToHost));
        if (abs_max <= threshold)
        {
            // We've reached an approximate solution within threshold
            out_x_vec.resize(size);
            verify(cudaMemcpy(out_x_vec.data(), d_x_vec, size * sizeof(float), cudaMemcpyDeviceToHost));
            return true;
        }

#if !USE_FUSED_KERNELS
        parallel_dot<<<LAUNCH_DOMAIN>>>(d_new_rdotr, d_r_vec, d_r_vec);

        // NOTE: `d_neg_alpha` is used as a dummy destination here
        scalar_div<<<1, 1>>>(d_beta, d_neg_alpha, d_new_rdotr, d_rdotr);
        verify(cudaMemcpyAsync(d_rdotr, d_new_rdotr, sizeof(float), cudaMemcpyDeviceToDevice));

        parallel_axby<<<LAUNCH_DOMAIN>>>(d_p_vec, d_r_vec, d_one, d_p_vec, d_beta);
#endif // USE_FUSED_KERNELS
    }

    return false;
}



