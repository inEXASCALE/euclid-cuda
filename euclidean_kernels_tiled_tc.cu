#include "euclidean_kernels_tiled.cuh"

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cstdint>
#include <type_traits>
#include <algorithm>

using namespace nvcuda;

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT32(x) TORCH_CHECK(x.scalar_type() == at::kFloat, #x " must be float32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT32(x)

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// Keep shared memory low: K-slice only (no MAX_DIM full tile in smem)
constexpr int BK = 16; // must be multiple of 16 for WMMA

template <typename T>
__device__ __forceinline__ float to_float(T v);

template <>
__device__ __forceinline__ float to_float<__half>(__half v) {
    return __half2float(v);
}

template <>
__device__ __forceinline__ float to_float<nv_bfloat16>(nv_bfloat16 v) {
#if __CUDA_ARCH__ >= 800
    return __bfloat162float(v);
#else
    return 0.0f;
#endif
}

template <>
__device__ __forceinline__ float to_float<float>(float v) {
    return v;
}

template <typename T>
__device__ __forceinline__ T from_float(float v);

template <>
__device__ __forceinline__ __half from_float<__half>(float v) {
    return __float2half(v);
}

template <>
__device__ __forceinline__ nv_bfloat16 from_float<nv_bfloat16>(float v) {
#if __CUDA_ARCH__ >= 800
    return __float2bfloat16(v);
#else
    return nv_bfloat16{};
#endif
}

template <>
__device__ __forceinline__ float from_float<float>(float v) {
    return v;
}

// -----------------------------
// Low-shared-memory TensorCore kernel
// - no P_smem[TILE][MAX_DIM], no C_smem[TILE][MAX_DIM]
// - no S_tile[TILE][TILE], no idx_list[]
// - compute one output element per thread
// - if fallback needed, recompute immediately in high precision
// -----------------------------
template <typename TCType, typename FallbackType, int TILE_N, int TILE_M>
__global__ void euclidean_tensorcore_generic_kernel(
    const float* __restrict__ P,   // [N, D]
    const float* __restrict__ C,   // [M, D]
    const float* __restrict__ P2,  // [N]
    const float* __restrict__ C2,  // [M]
    float* __restrict__ Out,       // [N, M]
    int N, int M, int D,
    float kappa,
    float u_low
) {
    const int tile_i = blockIdx.y;
    const int tile_j = blockIdx.x;
    const int i_start = tile_i * TILE_N;
    const int j_start = tile_j * TILE_M;

    const int tid = threadIdx.x;
    const int lane = tid & 31;      // thread in warp
    const int warp_id = tid >> 5;   // warp in block

    // We map 4 warps to 4 WMMA 16x16 subtiles in 32x32 tile:
    // warp 0 -> (0,0), warp 1 -> (0,16), warp 2 -> (16,0), warp 3 -> (16,16)
    // So this kernel is for TILE_N=TILE_M=32 with block=128 threads (4 warps).
    static_assert(TILE_N == 32 && TILE_M == 32, "This kernel currently supports tile 32x32 only.");

    if (warp_id >= 4) return;

    const int subtile_row = (warp_id / 2) * 16; // 0 or 16
    const int subtile_col = (warp_id % 2) * 16; // 0 or 16

    // shared memory for one K-slice only
    __shared__ TCType A_smem[32][BK]; // [TILE_N][BK]
    __shared__ TCType B_smem[32][BK]; // [TILE_M][BK]

    // WMMA accumulator for this warp's 16x16 subtile
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    // K loop by BK=16
    for (int k0 = 0; k0 < D; k0 += BK) {
        // cooperative load A_smem/B_smem
        // each block has 128 threads; load 32*BK + 32*BK elements
        for (int idx = tid; idx < 32 * BK; idx += blockDim.x) {
            int r = idx / BK;
            int c = idx % BK;

            int gi = i_start + r;
            int gk = k0 + c;

            float v = 0.0f;
            if (gi < N && gk < D) v = P[gi * D + gk];
            A_smem[r][c] = from_float<TCType>(v);
        }

        for (int idx = tid; idx < 32 * BK; idx += blockDim.x) {
            int r = idx / BK; // row in C tile (j_local)
            int c = idx % BK; // k in this slice

            int gj = j_start + r;
            int gk = k0 + c;

            float v = 0.0f;
            if (gj < M && gk < D) v = C[gj * D + gk];
            B_smem[r][c] = from_float<TCType>(v);
        }

        __syncthreads();

        if constexpr (std::is_same_v<TCType, __half>) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> b_frag;

            wmma::load_matrix_sync(a_frag, &A_smem[subtile_row][0], BK);
            wmma::load_matrix_sync(b_frag, &B_smem[subtile_col][0], BK);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        } else if constexpr (std::is_same_v<TCType, nv_bfloat16>) {
#if __CUDA_ARCH__ >= 800
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, nv_bfloat16, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, nv_bfloat16, wmma::col_major> b_frag;

            wmma::load_matrix_sync(a_frag, &A_smem[subtile_row][0], BK);
            wmma::load_matrix_sync(b_frag, &B_smem[subtile_col][0], BK);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
#endif
        }

        __syncthreads();
    }

    // Each warp stores its 16x16 accumulator to registers-per-thread path:
    // easiest robust way: store to small shared tile then each thread handles one element.
    __shared__ float C_smem[32][32];
    wmma::store_matrix_sync(&C_smem[subtile_row][subtile_col], c_frag, 32, wmma::mem_row_major);
    __syncthreads();

    // One thread computes one output element in the 32x32 tile
    int local_idx = tid; // 0..127
    // cover 1024 elements with stride
    const int r = 2 * D + 4;
    const float gamma_r = r * u_low;

    for (int linear = local_idx; linear < 32 * 32; linear += blockDim.x) {
        int i_local = linear / 32;
        int j_local = linear % 32;

        int i_global = i_start + i_local;
        int j_global = j_start + j_local;

        if (i_global >= N || j_global >= M) continue;

        float s = C_smem[i_local][j_local];
        float pp = P2[i_global];
        float cc = C2[j_global];

        float d = pp - 2.0f * s + cc;
        d = fmaxf(d, 0.0f);

        float eps_floor = kappa * gamma_r * (pp + cc);

        if (d <= eps_floor) {
            // immediate fallback recompute in high precision
            FallbackType sum = (FallbackType)0;
            for (int k = 0; k < D; ++k) {
                FallbackType pv = (FallbackType)P[i_global * D + k];
                FallbackType cv = (FallbackType)C[j_global * D + k];
                FallbackType diff = pv - cv;
                sum += diff * diff;
            }
            Out[i_global * M + j_global] = (float)sum;
        } else {
            Out[i_global * M + j_global] = d;
        }
    }
}

template <typename TCType, typename FallbackType>
static torch::Tensor launch_tensorcore_kernel(
    torch::Tensor P,
    torch::Tensor C,
    float kappa,
    int tile_size,
    float u_low
) {
    CHECK_INPUT(P);
    CHECK_INPUT(C);
    TORCH_CHECK(P.dim() == 2 && C.dim() == 2, "P and C must be 2D");
    TORCH_CHECK(P.size(1) == C.size(1), "dimension mismatch");

    int device = P.get_device();
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, device);
    TORCH_CHECK(prop.major >= 7, "TensorCore kernel requires sm70+");

    if constexpr (std::is_same_v<TCType, nv_bfloat16>) {
        TORCH_CHECK(prop.major >= 8, "BF16 TensorCore kernel requires sm80+");
    }

    const int64_t N64 = P.size(0);
    const int64_t M64 = C.size(0);
    const int64_t D64 = P.size(1);
    TORCH_CHECK(N64 <= INT_MAX && M64 <= INT_MAX && D64 <= INT_MAX, "shape too large");

    const int N = (int)N64;
    const int M = (int)M64;
    const int D = (int)D64;

    // fixed to 32 for low shared memory & robust mapping
    if (tile_size <= 0) tile_size = 32;
    TORCH_CHECK(tile_size == 32, "This fixed kernel supports tile_size=32 only");
    TORCH_CHECK((D % 16) == 0, "For WMMA path, D must be multiple of 16 (pad input if needed)");

    auto P2 = (P * P).sum(1);
    auto C2 = (C * C).sum(1);
    auto Out = torch::empty({N, M}, P.options().dtype(torch::kFloat32));

    dim3 grid((M + 31) / 32, (N + 31) / 32);
    dim3 block(128); // 4 warps

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    euclidean_tensorcore_generic_kernel<TCType, FallbackType, 32, 32>
        <<<grid, block, 0, stream>>>(
            P.data_ptr<float>(), C.data_ptr<float>(),
            P2.data_ptr<float>(), C2.data_ptr<float>(),
            Out.data_ptr<float>(),
            N, M, D, kappa, u_low
        );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return Out;
}

// -----------------------------
// exported APIs
// -----------------------------
torch::Tensor pairwise_euclidean_tensorcore_fp16_fp64(
    torch::Tensor P, torch::Tensor C, float kappa, int tile_size
) {
    constexpr float u_fp16 = 4.8828125e-4f; // 2^-11
    return launch_tensorcore_kernel<__half, double>(P, C, kappa, tile_size, u_fp16);
}

torch::Tensor pairwise_euclidean_tensorcore_fp16_fp32(
    torch::Tensor P, torch::Tensor C, float kappa, int tile_size
) {
    constexpr float u_fp16 = 4.8828125e-4f;
    return launch_tensorcore_kernel<__half, float>(P, C, kappa, tile_size, u_fp16);
}

torch::Tensor pairwise_euclidean_tensorcore_bf16_fp64(
    torch::Tensor P, torch::Tensor C, float kappa, int tile_size
) {
    constexpr float u_bf16 = 3.90625e-3f; // 2^-8
    return launch_tensorcore_kernel<nv_bfloat16, double>(P, C, kappa, tile_size, u_bf16);
}

torch::Tensor pairwise_euclidean_tensorcore_bf16_fp32(
    torch::Tensor P, torch::Tensor C, float kappa, int tile_size
) {
    constexpr float u_bf16 = 3.90625e-3f;
    return launch_tensorcore_kernel<nv_bfloat16, float>(P, C, kappa, tile_size, u_bf16);
}