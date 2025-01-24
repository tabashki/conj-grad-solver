#include "parallel.h"
#include "utils.h"

#include <iostream>
#include <cassert>
#include <cooperative_groups.h>

using std::vector;
namespace cg = cooperative_groups;


constexpr int BLOCK_SIZE = 128; // To avoid limiting the grid to too few SMs
constexpr int RDOTR_HISTORY_SIZE = 4;
enum ScalarIndex {
    S_ITER_NUM,
    S_ALPHA,
    S_NEG_ALPHA,
    S_BETA,
    S_P_DOT_AP,
    S_ABS_MAX,
    SCALAR_COUNT,
};
static_assert(SCALAR_COUNT >= RDOTR_HISTORY_SIZE);


__constant__ int d_vec_size;
__constant__ int d_max_iterations;
__constant__ float d_threshold;


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

//
// Fused Kernels
//

__global__ void fused_beta_matmul_dot(float *matmul_dest, float* scalars, const float* rdotr_hist, float* p_vec, const float* r_vec)
{
    const int global_i = blockDim.x * blockIdx.x + threadIdx.x;
    const int lane_i = threadIdx.x & 0x1F;
    const int i = threadIdx.x + 1;

    __shared__ int k;
    __shared__ float p_cache[BLOCK_SIZE + 2];
    __shared__ float dot_accum;
    __shared__ float beta;

    if (threadIdx.x == 0)
    {
        k = scalars[S_ITER_NUM];
        dot_accum = 0;
        beta = rdotr_hist[1] / rdotr_hist[2];

        const int li = global_i - 1;
        const int ri = global_i + BLOCK_SIZE;
        if (k > 1)
        {
            p_cache[0] = (li < 0) ? 0 : (p_vec[li] * beta + r_vec[li]);
            p_cache[BLOCK_SIZE+1] = (ri < d_vec_size) ? (p_vec[ri] * beta + r_vec[ri]) : 0;
        }
        else
        {
            p_cache[0] = (li < 0) ? 0 : p_vec[li];
            p_cache[BLOCK_SIZE+1] = (ri < d_vec_size) ? p_vec[ri] : 0;
        }
    }
    __syncthreads();

    // Result value for the matrix multiplication
    float result = 0;

    if (global_i < d_vec_size)
    {
        // Calculate new p_vec value and store it in shared cache,
        // as well as write back to the global buffer
        float new_p = p_vec[global_i];
        if (k > 1)
        {
            new_p = new_p * beta + r_vec[global_i];
            p_vec[global_i] = new_p;
        }
        p_cache[i] = new_p;
        result = new_p * -2;
    }
    else
    {
        p_cache[i] = 0;
    }

    __syncthreads();
    
    if (global_i < d_vec_size)
    {
        // Add in neighboring diagonal values
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
        atomicAdd(&scalars[S_P_DOT_AP], dot_accum);
    }
}

__global__ void fused_vecmul_absmax_dot(float *scalars, float* rdotr_hist,
                                        float* x_vec, float* r_vec, const float* p_vec, const float* a_mul_p)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    const int lane_i = threadIdx.x & 0x1F;

    __shared__ float alpha;
    __shared__ float dot_accum;
    __shared__ float max_accum;

    if (threadIdx.x == 0)
    {
        alpha = rdotr_hist[1] / scalars[S_P_DOT_AP];
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
        atomicAdd(&rdotr_hist[0], dot_accum);
        atomicAbsMaxFloat(&scalars[S_ABS_MAX], max_accum);
    }
}

__global__ void fused_update_scalars(float* scalars, float* rdotr_hist)
{
    const int i = threadIdx.x;

    __shared__ float history[RDOTR_HISTORY_SIZE];

    if (i == 0)
    {
        scalars[S_ITER_NUM] += 1;
    }
    else if (i < SCALAR_COUNT)
    {
        scalars[i] = 0;
    }
    if (i < RDOTR_HISTORY_SIZE)
    {
        history[i] = rdotr_hist[i];
        __syncthreads();

        if (i < (RDOTR_HISTORY_SIZE - 1))
        {
            rdotr_hist[i+1] = history[i];
        }
        rdotr_hist[0] = 0;
    }
}

//
// Uber-Kernel Implementation
//

__global__ void uber_coop_kernel(float *scalars, float* rdotr_hist, float* p_vec, float* r_vec, float* x_vec)
{
    const int gi = blockDim.x * blockIdx.x + threadIdx.x;
    const int lane = threadIdx.x & 0x1F;
    const int pi = threadIdx.x + 1;
    const int tid = threadIdx.x;

    __shared__ float p_cache[BLOCK_SIZE + 2];
    __shared__ float r_cache[BLOCK_SIZE];
    __shared__ float x_cache[BLOCK_SIZE];
    __shared__ float a_mul_p[BLOCK_SIZE];
    __shared__ float p_dot_ap;
    __shared__ float r_dot_r;
    __shared__ float prev_r_dot_r;
    __shared__ float abs_max;
    __shared__ float alpha;
    __shared__ float beta;

    if (tid == 0)
    {
        // Shared memory init
        p_dot_ap = 0;
        r_dot_r = rdotr_hist[0];
        prev_r_dot_r = rdotr_hist[1];
        abs_max = 0;
        alpha = 0;
        beta = 0;
    }

    if (gi < d_vec_size)
    {
        // Calculate new p_vec value and store it in shared cache,
        // as well as write back to the global buffer
        float new_p = p_vec[gi];
        p_cache[pi] = new_p;
        // mm = new_p * -2;

        r_cache[tid] = r_vec[gi];
        x_cache[tid] = x_vec[gi];
    }
    else
    {
        p_cache[pi] = 0;
        r_cache[tid] = 0;
        x_cache[tid] = 0;
    }

    auto grid = cg::this_grid();

    for (int k = 0; k < d_max_iterations; k++)
    {
        grid.sync();
    
        if (tid == 0)
        {
            prev_r_dot_r = r_dot_r;
            r_dot_r = 0;
            p_dot_ap = 0;
            abs_max = 0;

            const int li = gi - 1;
            const int ri = gi + BLOCK_SIZE;
            p_cache[0] = (li < 0) ? 0 : p_vec[li];
            p_cache[BLOCK_SIZE+1] = (ri < d_vec_size) ? p_vec[ri] : 0;
        }
        __syncthreads();

        if (gi == 0)
        {
            scalars[S_ITER_NUM] = k;
            scalars[S_ABS_MAX] = 0;
            scalars[S_P_DOT_AP] = 0;

            rdotr_hist[0] = 0;
            rdotr_hist[1] = prev_r_dot_r;
        }

        grid.sync();

        float mm = 0;
        if (gi < d_vec_size)
        {
            mm = p_cache[pi] * -2;
            // Add in neighboring diagonal values
            mm += p_cache[pi-1];
            mm += p_cache[pi+1];
            a_mul_p[tid] = mm;
        }

        // Fused wave-reduce dot product with p_vec
        mm *= p_cache[pi];

        mm += __shfl_down_sync(WARP_MASK, mm, 16);
        mm += __shfl_down_sync(WARP_MASK, mm, 8);
        mm += __shfl_down_sync(WARP_MASK, mm, 4);
        mm += __shfl_down_sync(WARP_MASK, mm, 2);
        mm += __shfl_down_sync(WARP_MASK, mm, 1);

        if (lane == 0)
        {
            atomicAdd(&p_dot_ap, mm);
        }
        __syncthreads();
        
        if (tid == 0)
        {
            atomicAdd(&scalars[S_P_DOT_AP], p_dot_ap);
        }
        __syncthreads();

        // Force a device-wide sync, after which the dot product value
        // should have been accumulated by the atomic adds
        grid.sync();

        if (tid == 0)
        {
            p_dot_ap = scalars[S_P_DOT_AP];
            alpha = prev_r_dot_r / p_dot_ap;
        }
        __syncthreads();

        float r = 0;
        if (gi < d_vec_size)
        {    
            x_cache[tid] = x_cache[tid] + p_cache[pi] * alpha;

            r = r_cache[tid] - a_mul_p[tid] * alpha;
            r_cache[tid] = r;
        }

        float rm = abs(r);
        float rd = r * r;

        rm = max(rm, __shfl_down_sync(WARP_MASK, rm, 16));
        rd += __shfl_down_sync(WARP_MASK, rd, 16);

        rm = max(rm, __shfl_down_sync(WARP_MASK, rm, 8));
        rd += __shfl_down_sync(WARP_MASK, rd, 8);

        rm = max(rm, __shfl_down_sync(WARP_MASK, rm, 4));
        rd += __shfl_down_sync(WARP_MASK, rd, 4);

        rm = max(rm, __shfl_down_sync(WARP_MASK, rm, 2));
        rd += __shfl_down_sync(WARP_MASK, rd, 2);

        rm = max(rm, __shfl_down_sync(WARP_MASK, rm, 1));
        rd += __shfl_down_sync(WARP_MASK, rd, 1);

        if (lane == 0)
        {
            atomicAdd(&r_dot_r, rd);
            atomicAbsMaxFloat(&abs_max, rm);
        }
        __syncthreads();

        if (tid == 0)
        {
            atomicAdd(&rdotr_hist[0], r_dot_r);
            atomicAbsMaxFloat(&scalars[S_ABS_MAX], abs_max);
        }
        
        grid.sync();

        if (tid == 0)
        {
            r_dot_r = rdotr_hist[0];
            abs_max = scalars[S_ABS_MAX];
            beta = r_dot_r / prev_r_dot_r;
        }
        __syncthreads();

        if (abs_max <= d_threshold)
        {
            break;
        }

        if (gi < d_vec_size)
        {
            const float p = r_cache[tid] + p_cache[pi] * beta; 
            p_cache[pi] = p;
            // Write-back p_vec so that boundary values can be read out by surrounding blocks
            p_vec[gi] = p;
        }
    }

    // Write-back to global buffers
    if (gi < d_vec_size)
    {
        p_vec[gi] = p_cache[pi];
        r_vec[gi] = r_cache[tid];
        x_vec[gi] = x_cache[tid];
    }
}

//
// Host-side housekeeping
//

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

#define USE_UBER_KERNEL   1
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
    verify(cudaMemcpyToSymbol(d_max_iterations, &max_iters, sizeof(d_max_iterations)));
    verify(cudaMemcpyToSymbol(d_threshold, &threshold, sizeof(d_threshold)));

    constexpr int device_id = 0;
    int supports_coop_launch = 0;
    cudaDeviceGetAttribute(&supports_coop_launch, cudaDevAttrCooperativeLaunch, device_id);
    std::cout << "Device " << device_id << " Supports Cooperative Launch: " << (supports_coop_launch ? "Yes" : "No") << std::endl;
    if (!supports_coop_launch)
    {
        std::cout << "Cannot continue without CUDA co-op launch support!" << std::endl;
        return false;
    }

    cudaDeviceProp device_props;
    cudaGetDeviceProperties(&device_props, device_id);
    int max_block_occupancy = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_block_occupancy, uber_coop_kernel, BLOCK_SIZE, 0);
    max_block_occupancy *= device_props.multiProcessorCount;
    
    const int blocks = div_round_up(size, BLOCK_SIZE);
    std::cout << "Device " << device_id << " supports max: " << max_block_occupancy << " blocks launched" << std::endl;
    if (max_block_occupancy < blocks)
    {
        std::cout << "Device " << device_id << " doesn't support required number of blocks: " << blocks << std::endl;
        return false;
    }

    cudaEvent_t start_event;
    cudaEvent_t end_event;

    verify(cudaEventCreate(&start_event));
    verify(cudaEventCreate(&end_event));

    verify(cudaEventRecord(start_event));

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

    float* d_scalars;
    float* d_rdotr_history;

    verify(cudaMalloc(&d_one, sizeof(float)));
    verify(cudaMalloc(&d_neg_one, sizeof(float)));

    verify(cudaMalloc(&d_scalars, sizeof(float) * SCALAR_COUNT));
    verify(cudaMalloc(&d_rdotr_history, sizeof(float) * RDOTR_HISTORY_SIZE));
    verify(cudaMemset(d_scalars, 0, sizeof(float) * SCALAR_COUNT));
    verify(cudaMemset(d_rdotr_history, 0, sizeof(float) * RDOTR_HISTORY_SIZE));

    const float one = 1.0;
    const float neg_one = -1.0;

    verify(cudaMemset(d_x_vec, 0, size * sizeof(float)));
    verify(cudaMemset(d_a_mul_p, 0, size * sizeof(float)));

    verify(cudaMemcpy(d_one, &one, sizeof(float), cudaMemcpyHostToDevice));
    verify(cudaMemcpy(d_rdotr_history + 1, &one, sizeof(float), cudaMemcpyHostToDevice));
    verify(cudaMemcpy(d_rdotr_history + 2, &one, sizeof(float), cudaMemcpyHostToDevice));

    verify(cudaMemcpy(d_neg_one, &neg_one, sizeof(float), cudaMemcpyHostToDevice));
    verify(cudaMemcpy(d_r_vec, in_b_vec.data(), size * sizeof(float), cudaMemcpyHostToDevice));
    verify(cudaMemcpy(d_p_vec, d_r_vec, size * sizeof(float), cudaMemcpyDeviceToDevice));

    parallel_dot<<<LAUNCH_DOMAIN>>>(d_rdotr_history, d_r_vec, d_r_vec);

    float execution_time_ms = 0;

#if USE_UBER_KERNEL
    // Uber kernel only launches a single kernel that internally loops
    { 
        verify(cudaMemsetAsync(d_scalars, 0, sizeof(float) * SCALAR_COUNT));

        void *kernelArgs[] = { &d_scalars, &d_rdotr_history, &d_p_vec, &d_r_vec, &d_x_vec, };
        dim3 block_dim(BLOCK_SIZE, 1, 1);
        dim3 grid_dim(blocks, 1, 1);
        cudaLaunchCooperativeKernel((void*)uber_coop_kernel, grid_dim, block_dim, kernelArgs);  
#else
    constexpr int BATCH_SIZE = 10;

    for (int k = 0; k < max_iters;)
    {
        for (int b = 0; b < BATCH_SIZE; b++)
        {

#if USE_FUSED_KERNELS
            fused_update_scalars<<<1, SCALAR_COUNT>>>(d_scalars, d_rdotr_history);
            fused_beta_matmul_dot<<<LAUNCH_DOMAIN>>>(d_a_mul_p, d_scalars, d_rdotr_history, d_p_vec, d_r_vec);
            fused_vecmul_absmax_dot<<<LAUNCH_DOMAIN>>>(d_scalars, d_rdotr_history, d_x_vec, d_r_vec, d_p_vec, d_a_mul_p);
#else
            verify(cudaMemsetAsync(d_scalars, 0, sizeof(float) * (SCALAR_COUNT)));

            parallel_sp_matmul<<<LAUNCH_DOMAIN>>>(d_a_mul_p, d_p_vec); 

            parallel_dot<<<LAUNCH_DOMAIN>>>(d_scalars+S_P_DOT_AP, d_p_vec, d_a_mul_p);
            scalar_div<<<1, 1>>>(d_scalars+S_ALPHA, d_scalars+S_NEG_ALPHA, d_rdotr_history, d_scalars+S_P_DOT_AP);

            parallel_axby<<<LAUNCH_DOMAIN>>>(d_x_vec, d_x_vec, d_one, d_p_vec, d_scalars+S_ALPHA);
            parallel_axby<<<LAUNCH_DOMAIN>>>(d_r_vec, d_r_vec, d_one, d_a_mul_p, d_scalars+S_NEG_ALPHA);

            parallel_absmax<<<LAUNCH_DOMAIN>>>(d_scalars+S_ABS_MAX, d_r_vec);

            verify(cudaMemcpyAsync(d_rdotr_history+1, d_rdotr_history, sizeof(float), cudaMemcpyDeviceToDevice));
            verify(cudaMemsetAsync(d_rdotr_history, 0, sizeof(float)));
            parallel_dot<<<LAUNCH_DOMAIN>>>(d_rdotr_history, d_r_vec, d_r_vec);

            // NOTE: `S_NEG_ALPHA` is used as a dummy destination here
            scalar_div<<<1, 1>>>(d_scalars+S_BETA, d_scalars+S_NEG_ALPHA, d_rdotr_history, d_rdotr_history+1);

            parallel_axby<<<LAUNCH_DOMAIN>>>(d_p_vec, d_r_vec, d_one, d_p_vec, d_scalars+S_BETA);
        }
        k += BATCH_SIZE;
#endif // USE_FUSED_KERNELS
#endif // USE_UBER_KERNEL

        float iters = 0;
        float abs_max = 0;
        verify(cudaMemcpy(&iters, d_scalars+S_ITER_NUM, sizeof(abs_max), cudaMemcpyDeviceToHost));
        verify(cudaMemcpy(&abs_max, d_scalars+S_ABS_MAX, sizeof(abs_max), cudaMemcpyDeviceToHost));
        if (abs_max <= threshold)
        {
            cudaEventRecord(end_event);

            // We've reached an approximate solution within threshold
            out_x_vec.resize(size);
            verify(cudaMemcpy(out_x_vec.data(), d_x_vec, size * sizeof(float), cudaMemcpyDeviceToHost));

            verify(cudaEventElapsedTime(&execution_time_ms, start_event, end_event));
            std::cout << "GPU execution took: " << execution_time_ms << " ms" << std::endl;

            return true;
        }
    }

    return false;
}



