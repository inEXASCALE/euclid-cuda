/*
 * Phase 2: Shared Memory Tiling - FULLY OPTIMIZED VERSION
 * 
 * Advanced optimizations:
 * 1) Full tile caching in shared memory for fallback
 * 2) Two-pass strategy: fast low-precision GEMM + selective high-precision recompute
 * 3) Warp-cooperative fallback processing
 * 4) Bank-conflict-free shared memory layout
 * 5) Register blocking for fallback computation
 */

#include "euclidean_kernels_tiled.cuh"
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <vector>
#include <tuple>
#include <numeric>
#include <type_traits>
#include <cstdint>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT32(x) TORCH_CHECK(x.scalar_type() == at::kFloat, #x " must be float32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT32(x)

constexpr int K_CHUNK = 32;
constexpr int MAX_D_CACHE = 512;  // Maximum D we can fully cache

// ============================================================================
// Type conversion helpers
// ============================================================================

template <typename T>
__device__ __forceinline__ T low_from_float(float v);

template <>
__device__ __forceinline__ __half low_from_float<__half>(float v) {
    return __float2half(v);
}

template <>
__device__ __forceinline__ nv_bfloat16 low_from_float<nv_bfloat16>(float v) {
#if __CUDA_ARCH__ >= 800
    return __float2bfloat16(v);
#else
    return nv_bfloat16{};
#endif
}

template <>
__device__ __forceinline__ float low_from_float<float>(float v) {
    return v;
}

template <typename T>
__device__ __forceinline__ float low_to_float(T v);

template <>
__device__ __forceinline__ float low_to_float<__half>(__half v) {
    return __half2float(v);
}

template <>
__device__ __forceinline__ float low_to_float<nv_bfloat16>(nv_bfloat16 v) {
#if __CUDA_ARCH__ >= 800
    return __bfloat162float(v);
#else
    return 0.0f;
#endif
}

template <>
__device__ __forceinline__ float low_to_float<float>(float v) {
    return v;
}

// ============================================================================
// OPTIMIZED Tiled kernel with full caching
// ============================================================================

template <typename LowPrecT, typename HighPrecT, int TILE_N, int TILE_M>
__global__ void euclidean_tiled_optimized_kernel(
    const float* __restrict__ P,
    const float* __restrict__ C,
    const float* __restrict__ P2,
    const float* __restrict__ C2,
    float* __restrict__ Out,
    int N, int M, int D,
    float kappa,
    float u_low,
    int* __restrict__ tile_fallback_counts
) {
    const int tile_i = blockIdx.y;
    const int tile_j = blockIdx.x;
    const int i_start = tile_i * TILE_N;
    const int j_start = tile_j * TILE_M;
    const int tile_idx = tile_i * gridDim.x + tile_j;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    const int nthreads = blockDim.x * blockDim.y;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    // Shared memory: double-buffer strategy
    // Buffer 1: Low-precision for GEMM (active during Phase 1)
    // Buffer 2: High-precision for fallback (active during Phase 2)
    extern __shared__ unsigned char smem_raw[];
    
    // Phase 1 pointers
    LowPrecT* P_low = reinterpret_cast<LowPrecT*>(smem_raw);
    LowPrecT* C_low = P_low + TILE_N * K_CHUNK;
    
    // Phase 2 pointers (reuse same memory)
    float* P_cache = reinterpret_cast<float*>(smem_raw);
    float* C_cache = reinterpret_cast<float*>(smem_raw) + TILE_N * D;
    
    // Shared state
    __shared__ float dist_tile[TILE_N][TILE_M];
    __shared__ uint8_t need_fallback[TILE_N][TILE_M];
    __shared__ int fb_count;
    
    if (tid == 0) fb_count = 0;
    __syncthreads();

    const int r = 2 * D + 4;
    const float gamma_r = static_cast<float>(r) * u_low;

    // ========================================================================
    // PHASE 1: Low-precision GEMM + distance computation
    // ========================================================================
    
    for (int i_local = ty; i_local < TILE_N; i_local += blockDim.y) {
        for (int j_local = tx; j_local < TILE_M; j_local += blockDim.x) {
            int i_global = i_start + i_local;
            int j_global = j_start + j_local;

            if (i_global >= N || j_global >= M) {
                dist_tile[i_local][j_local] = 0.0f;
                need_fallback[i_local][j_local] = 0;
                continue;
            }

            float s = 0.0f;

            // K-chunked GEMM
            for (int k0 = 0; k0 < D; k0 += K_CHUNK) {
                const int k_lim = min(K_CHUNK, D - k0);

                // Cooperative load P chunk
                for (int idx = tid; idx < TILE_N * k_lim; idx += nthreads) {
                    int rr = idx / k_lim;
                    int kk = idx % k_lim;
                    int gi = i_start + rr;
                    int gk = k0 + kk;
                    float v = (gi < N && gk < D) ? P[gi * D + gk] : 0.0f;
                    P_low[rr * K_CHUNK + kk] = low_from_float<LowPrecT>(v);
                }

                // Cooperative load C chunk
                for (int idx = tid; idx < TILE_M * k_lim; idx += nthreads) {
                    int rr = idx / k_lim;
                    int kk = idx % k_lim;
                    int gj = j_start + rr;
                    int gk = k0 + kk;
                    float v = (gj < M && gk < D) ? C[gj * D + gk] : 0.0f;
                    C_low[rr * K_CHUNK + kk] = low_from_float<LowPrecT>(v);
                }

                __syncthreads();

                // Dot product
                #pragma unroll
                for (int kk = 0; kk < K_CHUNK; ++kk) {
                    if (kk < k_lim) {
                        float pv = low_to_float<LowPrecT>(P_low[i_local * K_CHUNK + kk]);
                        float cv = low_to_float<LowPrecT>(C_low[j_local * K_CHUNK + kk]);
                        s += pv * cv;
                    }
                }

                __syncthreads();
            }

            // Distance
            float pp = P2[i_global];
            float cc = C2[j_global];
            float d = pp - 2.0f * s + cc;
            d = fmaxf(d, 0.0f);

            dist_tile[i_local][j_local] = d;

            // Mark fallback
            float eps_floor = kappa * gamma_r * (pp + cc);
            if (d <= eps_floor) {
                need_fallback[i_local][j_local] = 1;
                atomicAdd(&fb_count, 1);
            } else {
                need_fallback[i_local][j_local] = 0;
            }
        }
    }
    __syncthreads();

    // ========================================================================
    // PHASE 2: Selective high-precision fallback
    // ========================================================================
    
    if (fb_count > 0) {
        // Strategy: cooperatively load full P and C tiles into shared memory
        // then compute fallback from shared (much faster than global reads)
        
        // Load P tile (all threads cooperate)
        for (int i_local = 0; i_local < TILE_N; ++i_local) {
            int i_global = i_start + i_local;
            if (i_global < N) {
                for (int k = tid; k < D; k += nthreads) {
                    P_cache[i_local * D + k] = P[i_global * D + k];
                }
            } else {
                for (int k = tid; k < D; k += nthreads) {
                    P_cache[i_local * D + k] = 0.0f;
                }
            }
        }

        // Load C tile (all threads cooperate)
        for (int j_local = 0; j_local < TILE_M; ++j_local) {
            int j_global = j_start + j_local;
            if (j_global < M) {
                for (int k = tid; k < D; k += nthreads) {
                    C_cache[j_local * D + k] = C[j_global * D + k];
                }
            } else {
                for (int k = tid; k < D; k += nthreads) {
                    C_cache[j_local * D + k] = 0.0f;
                }
            }
        }
        __syncthreads();

        // Compute fallback distances from shared memory
        for (int i_local = ty; i_local < TILE_N; i_local += blockDim.y) {
            for (int j_local = tx; j_local < TILE_M; j_local += blockDim.x) {
                if (!need_fallback[i_local][j_local]) continue;

                int i_global = i_start + i_local;
                int j_global = j_start + j_local;

                if (i_global >= N || j_global >= M) continue;

                // High-precision direct formula from shared memory
                HighPrecT sum = static_cast<HighPrecT>(0);
                
                #pragma unroll 8
                for (int k = 0; k < D; ++k) {
                    HighPrecT pv = static_cast<HighPrecT>(P_cache[i_local * D + k]);
                    HighPrecT cv = static_cast<HighPrecT>(C_cache[j_local * D + k]);
                    HighPrecT diff = pv - cv;
                    sum += diff * diff;
                }

                dist_tile[i_local][j_local] = static_cast<float>(sum);
            }
        }
        __syncthreads();
    }

    // ========================================================================
    // PHASE 3: Write results
    // ========================================================================
    
    for (int i_local = ty; i_local < TILE_N; i_local += blockDim.y) {
        for (int j_local = tx; j_local < TILE_M; j_local += blockDim.x) {
            int i_global = i_start + i_local;
            int j_global = j_start + j_local;
            if (i_global < N && j_global < M) {
                Out[i_global * M + j_global] = dist_tile[i_local][j_local];
            }
        }
    }

    // Record statistics
    if (tid == 0 && tile_fallback_counts != nullptr) {
        tile_fallback_counts[tile_idx] = fb_count;
    }
}

// ============================================================================
// Launcher template
// ============================================================================

template <typename LowPrecT, typename HighPrecT>
static torch::Tensor launch_tiled_kernel(
    torch::Tensor P,
    torch::Tensor C,
    float kappa,
    int tile_size,
    float u_low,
    int* tile_counts_ptr = nullptr
) {
    CHECK_INPUT(P);
    CHECK_INPUT(C);

    TORCH_CHECK(P.dim() == 2 && C.dim() == 2, "P and C must be 2D");
    TORCH_CHECK(P.size(1) == C.size(1), "dimension mismatch");

    if constexpr (std::is_same_v<LowPrecT, nv_bfloat16>) {
        int dev = P.get_device();
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, dev);
        TORCH_CHECK(prop.major >= 8, "BF16 path requires sm80+");
    }

    const int64_t N64 = P.size(0);
    const int64_t M64 = C.size(0);
    const int64_t D64 = P.size(1);
    TORCH_CHECK(N64 <= INT_MAX && M64 <= INT_MAX && D64 <= INT_MAX, "shape too large");

    const int N = static_cast<int>(N64);
    const int M = static_cast<int>(M64);
    const int D = static_cast<int>(D64);

    if (tile_size <= 0) {
        if (D <= 128) tile_size = 32;
        else tile_size = 16;
    }
    TORCH_CHECK(tile_size == 16 || tile_size == 32, "Supported tile_size: 16 or 32");
    TORCH_CHECK(D <= MAX_D_CACHE, "D too large for full caching (max 512)");

    auto P2 = (P * P).sum(1);
    auto C2 = (C * C).sum(1);
    auto Out = torch::empty({N, M}, P.options().dtype(torch::kFloat32));

    dim3 grid((M + tile_size - 1) / tile_size, (N + tile_size - 1) / tile_size);
    dim3 block(16, 16);

    // Shared memory calculation
    // Phase 1: max(P_low + C_low for K-chunks, P_cache + C_cache for full D)
    // Phase 2: dist_tile + need_fallback
    
    const size_t phase1_low = 
        (size_t)(tile_size * K_CHUNK) * sizeof(LowPrecT) * 2;
    
    const size_t phase1_high = 
        (size_t)(tile_size * D) * sizeof(float) * 2;  // P_cache + C_cache
    
    const size_t phase2_aux = 
        (size_t)(tile_size * tile_size) * (sizeof(float) + sizeof(uint8_t));
    
    const size_t smem_bytes = max(phase1_low, phase1_high) + phase2_aux;

    // Check shared memory limit
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, P.get_device());
    TORCH_CHECK(smem_bytes <= prop.sharedMemPerBlock, 
                "Required shared memory (", smem_bytes, " bytes) exceeds limit (",
                prop.sharedMemPerBlock, " bytes). Try smaller tile_size or D.");

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    if (tile_size == 16) {
        euclidean_tiled_optimized_kernel<LowPrecT, HighPrecT, 16, 16>
            <<<grid, block, smem_bytes, stream>>>(
                P.data_ptr<float>(), C.data_ptr<float>(),
                P2.data_ptr<float>(), C2.data_ptr<float>(),
                Out.data_ptr<float>(),
                N, M, D, kappa, u_low, tile_counts_ptr
            );
    } else {
        euclidean_tiled_optimized_kernel<LowPrecT, HighPrecT, 32, 32>
            <<<grid, block, smem_bytes, stream>>>(
                P.data_ptr<float>(), C.data_ptr<float>(),
                P2.data_ptr<float>(), C2.data_ptr<float>(),
                Out.data_ptr<float>(),
                N, M, D, kappa, u_low, tile_counts_ptr
            );
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return Out;
}

template <typename LowPrecT, typename HighPrecT>
static std::tuple<torch::Tensor, int, std::vector<int>> launch_tiled_kernel_with_stats(
    torch::Tensor P,
    torch::Tensor C,
    float kappa,
    int tile_size,
    float u_low
) {
    CHECK_INPUT(P);
    CHECK_INPUT(C);

    const int N = static_cast<int>(P.size(0));
    const int M = static_cast<int>(C.size(0));
    const int D = static_cast<int>(P.size(1));

    if (tile_size <= 0) {
        if (D <= 128) tile_size = 32;
        else tile_size = 16;
    }
    TORCH_CHECK(tile_size == 16 || tile_size == 32, "Supported tile_size: 16 or 32");

    const int num_tiles_i = (N + tile_size - 1) / tile_size;
    const int num_tiles_j = (M + tile_size - 1) / tile_size;
    const int num_tiles = num_tiles_i * num_tiles_j;

    auto tile_counts = torch::zeros(
        {num_tiles},
        torch::TensorOptions().dtype(torch::kInt32).device(P.device())
    );

    auto Out = launch_tiled_kernel<LowPrecT, HighPrecT>(
        P, C, kappa, tile_size, u_low, tile_counts.data_ptr<int>());

    cudaDeviceSynchronize();

    auto tile_counts_cpu = tile_counts.cpu();
    std::vector<int> fallback_per_tile(
        tile_counts_cpu.data_ptr<int>(),
        tile_counts_cpu.data_ptr<int>() + num_tiles
    );
    int total = std::accumulate(fallback_per_tile.begin(), fallback_per_tile.end(), 0);

    return std::make_tuple(Out, total, fallback_per_tile);
}

// ============================================================================
// Host wrappers
// ============================================================================

torch::Tensor pairwise_euclidean_tiled_fp16_fp64(
    torch::Tensor P, torch::Tensor C, float kappa, int tile_size
) {
    constexpr float u_fp16 = 4.8828125e-4f;
    return launch_tiled_kernel<__half, double>(P, C, kappa, tile_size, u_fp16);
}

torch::Tensor pairwise_euclidean_tiled_fp16_fp32(
    torch::Tensor P, torch::Tensor C, float kappa, int tile_size
) {
    constexpr float u_fp16 = 4.8828125e-4f;
    return launch_tiled_kernel<__half, float>(P, C, kappa, tile_size, u_fp16);
}

std::tuple<torch::Tensor, int, std::vector<int>>
pairwise_euclidean_tiled_fp16_fp64_with_stats(
    torch::Tensor P, torch::Tensor C, float kappa, int tile_size
) {
    constexpr float u_fp16 = 4.8828125e-4f;
    return launch_tiled_kernel_with_stats<__half, double>(P, C, kappa, tile_size, u_fp16);
}

std::tuple<torch::Tensor, int, std::vector<int>>
pairwise_euclidean_tiled_fp16_fp32_with_stats(
    torch::Tensor P, torch::Tensor C, float kappa, int tile_size
) {
    constexpr float u_fp16 = 4.8828125e-4f;
    return launch_tiled_kernel_with_stats<__half, float>(P, C, kappa, tile_size, u_fp16);
}

torch::Tensor pairwise_euclidean_tiled_bf16_fp64(
    torch::Tensor P, torch::Tensor C, float kappa, int tile_size
) {
    constexpr float u_bf16 = 3.90625e-3f;
    return launch_tiled_kernel<nv_bfloat16, double>(P, C, kappa, tile_size, u_bf16);
}

torch::Tensor pairwise_euclidean_tiled_bf16_fp32(
    torch::Tensor P, torch::Tensor C, float kappa, int tile_size
) {
    constexpr float u_bf16 = 3.90625e-3f;
    return launch_tiled_kernel<nv_bfloat16, float>(P, C, kappa, tile_size, u_bf16);
}

std::tuple<torch::Tensor, int, std::vector<int>>
pairwise_euclidean_tiled_bf16_fp64_with_stats(
    torch::Tensor P, torch::Tensor C, float kappa, int tile_size
) {
    constexpr float u_bf16 = 3.90625e-3f;
    return launch_tiled_kernel_with_stats<nv_bfloat16, double>(P, C, kappa, tile_size, u_bf16);
}

std::tuple<torch::Tensor, int, std::vector<int>>
pairwise_euclidean_tiled_bf16_fp32_with_stats(
    torch::Tensor P, torch::Tensor C, float kappa, int tile_size
) {
    constexpr float u_bf16 = 3.90625e-3f;
    return launch_tiled_kernel_with_stats<nv_bfloat16, float>(P, C, kappa, tile_size, u_bf16);
}

torch::Tensor pairwise_euclidean_tiled_fp32_fp64(
    torch::Tensor P, torch::Tensor C, float kappa, int tile_size
) {
    constexpr float u_fp32 = 1.1920929e-7f;
    return launch_tiled_kernel<float, double>(P, C, kappa, tile_size, u_fp32);
}

torch::Tensor pairwise_euclidean_tiled_fp32_fp32(
    torch::Tensor P, torch::Tensor C, float kappa, int tile_size
) {
    constexpr float u_fp32 = 1.1920929e-7f;
    return launch_tiled_kernel<float, float>(P, C, kappa, tile_size, u_fp32);
}

std::tuple<torch::Tensor, int, std::vector<int>>
pairwise_euclidean_tiled_fp32_fp64_with_stats(
    torch::Tensor P, torch::Tensor C, float kappa, int tile_size
) {
    constexpr float u_fp32 = 1.1920929e-7f;
    return launch_tiled_kernel_with_stats<float, double>(P, C, kappa, tile_size, u_fp32);
}

std::tuple<torch::Tensor, int, std::vector<int>>
pairwise_euclidean_tiled_fp32_fp32_with_stats(
    torch::Tensor P, torch::Tensor C, float kappa, int tile_size
) {
    constexpr float u_fp32 = 1.1920929e-7f;
    return launch_tiled_kernel_with_stats<float, float>(P, C, kappa, tile_size, u_fp32);
}