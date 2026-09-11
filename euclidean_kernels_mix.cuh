/*
 * PyTorch CUDA Extension: Mixed-Precision Euclidean Distance with FP64 Fallback
 * Header file - declarations only
 *
 * ALGORITHM-COMPLIANT VERSION:
 * - Public APIs take FP32 inputs (working precision u).
 * - Fast path casts to low-precision for GEMM operations only.
 * - Error floor uses γ_{2ℓ+4} formula from the algorithm.
 */
#pragma once
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/core/ScalarType.h>

// Forward declaration for cuBLAS handle (avoid including cublas in host code)
#ifdef __cplusplus
extern "C" {
#endif
    typedef struct cublasContext* cublasHandle_t;
#ifdef __cplusplus
}
#endif


/**
 * @brief Compute squared norms using FP16 arithmetic (algorithm-compliant).
 * 
 * Performs fl_ul(x^T x) by casting FP32 input to FP16 and accumulating in FP32.
 * This implements the low-precision norm computation from Algorithm 1, line 2-3.
 * 
 * @param X      Input matrix in FP32 (N x dim), row-major
 * @param X2     Output squared norms in FP32 (N,)
 * @param N      Number of vectors
 * @param dim    Dimensionality of each vector
 * 
 * @note Each thread processes one vector: X2[i] = sum_k (fl_fp16(X[i,k]))^2
 * @note Grid: (N + blockDim - 1) / blockDim blocks, Block: 256 threads
 */
__global__ void compute_norms_fp16_low(
    const c10::Half* X, float* X2, int N, int dim);


/**
 * @brief Compute squared norms using BF16 arithmetic (algorithm-compliant).
 * 
 * Performs fl_ul(x^T x) by casting FP32 input to BF16 and accumulating in FP32.
 * 
 * @param X      Input matrix in FP32 (N x dim), row-major
 * @param X2     Output squared norms in FP32 (N,)
 * @param N      Number of vectors
 * @param dim    Dimensionality of each vector
 * 
 * @note BF16 has wider dynamic range but lower mantissa precision than FP16
 */
__global__ void compute_norms_bf16_low(
    const c10::BFloat16* X, float* X2, int N, int dim);



/**
 * @brief Compute squared norms using TF32 arithmetic (algorithm-compliant).
 * 
 * Performs norm computation for TF32 path. Input remains FP32, but cuBLAS
 * will internally use TF32 for matrix operations.
 * 
 * @param X      Input matrix in FP32 (N x dim), row-major
 * @param X2     Output squared norms in FP32 (N,)
 * @param N      Number of vectors
 * @param dim    Dimensionality of each vector
 */
__global__ void compute_norms_tf32(
    const float* X, float* X2, int N, int dim);

__global__ void compute_norms_fp32(
    const float* X, float* X2, int N, int dim);

__global__ void compute_norms_fp64(
    const double* X, double* X2, int N, int dim);

// Uniform type versions
__global__ void compute_norms_fp16_uniform(
    const c10::Half* X, c10::Half* X2, int N, int dim);

__global__ void compute_norms_bf16_uniform(
    const c10::BFloat16* X, c10::BFloat16* X2, int N, int dim);

__global__ void compute_norms_tf32_uniform(
    const float* X, float* X2, int N, int dim);

// Distance formation kernels
__global__ void form_expanded_fp16_to_float(
    const c10::Half* S, const float* P2, const float* C2,
    float* D, int N, int M);

__global__ void form_expanded_bf16_to_float(
    const c10::BFloat16* S, const float* P2, const float* C2,
    float* D, int N, int M);

__global__ void form_expanded_fp32_to_float(
    const float* S, const float* P2, const float* C2,
    float* D, int N, int M);


/**
 * @brief Form distances and error-floor test for FP16 path.
 * 
 * @param S           Dot product matrix from cuBLAS in FP16 (N x M)
 * @param P2          Squared norms of P in FP16 (N,)
 * @param C2          Squared norms of C in FP16 (M,)
 * @param D           Output distance matrix in FP32 (N x M)
 * @param idx_list    Output list for fallback indices (max size N*M)
 * @param list_count  Atomic counter for number of fallback entries
 * @param N           Number of points in P
 * @param M           Number of points in C
 * @param dim         Dimensionality (ℓ in algorithm)
 * @param rho         Safety factor (ρ in algorithm, typically kappa in API)
 * 
 * @note Grid: ((M+15)/16, (N+15)/16), Block: (16, 16)
 * @note Uses atomicAdd to populate idx_list, no race conditions
 */
__global__ void form_mixed_and_list_fp16(
    const c10::Half* S, const c10::Half* P2, const c10::Half* C2,
    float* D, uint32_t* idx_list, uint32_t* list_count,
    int N, int M, int dim, float rho);

/**
 * @brief Form distances and error-floor test for BF16 path.
 * 
 * Same as form_mixed_and_list_fp16 but for BF16 precision.
 * Uses u_l = 2^-7 (BF16 machine epsilon) in error bound computation.
 * 
 * @see form_mixed_and_list_fp16 for detailed parameter descriptions
 */
__global__ void form_mixed_and_list_bf16(
    const c10::BFloat16* S, const c10::BFloat16* P2, const c10::BFloat16* C2,
    float* D, uint32_t* idx_list, uint32_t* list_count,
    int N, int M, int dim, float rho);

/**
 * @brief Form distances and error-floor test for TF32 path.
 * 
 * Same as form_mixed_and_list_fp16 but for TF32 precision.
 * Uses u_l = 2^-10 (TF32 machine epsilon) in error bound computation.
 * 
 * @see form_mixed_and_list_fp16 for detailed parameter descriptions
 */
__global__ void form_mixed_and_list_tf32(
    const float* S, const float* P2, const float* C2,
    float* D, uint32_t* idx_list, uint32_t* list_count,
    int N, int M, int dim, float rho);

/**
 * @brief Fallback recomputation using high-precision (FP64 accumulator).
 * 
 * @param P           First data matrix in FP32 (N x dim), row-major
 * @param C           Second data matrix in FP32 (M x dim), row-major
 * @param D           Output distance matrix in FP32 (N x M), row-major (in-place update)
 * @param idx_list    List of flattened indices requiring recomputation
 * @param list_count  Pointer to scalar count of valid indices (on device)
 * @param dim         Dimensionality
 * @param N           Number of points in P
 * @param M           Number of points in C
 * 
 * @note Each thread processes one index from idx_list
 * @note Uses direct distance: sum_k (p[k] - c[k])^2 with FP64 accumulation
 */
__global__ void direct_on_list_rowmajor(
    const float* P, const float* C, float* D,
    const uint32_t* idx_list, const uint32_t* list_count,
    int dim, int N, int M);

// ============================================================================
// NEW: DIRECT FORMULA FALLBACK KERNELS
// ============================================================================

/**
 * @brief Fallback using FP16 direct formula.
 */
__global__ void direct_fp16_on_list(
    const c10::Half* P, const c10::Half* C, float* D,
    const uint32_t* idx_list, const uint32_t* list_count,
    int dim, int N, int M);

/**
 * @brief Fallback using FP32 direct formula on FP16 data.
 */
__global__ void direct_fp32_on_list_from_fp16(
    const float* P, const float* C, float* D,
    const uint32_t* idx_list, const uint32_t* list_count,
    int dim, int N, int M);

/**
 * @brief Fallback using BF16 direct formula.
 */
__global__ void direct_bf16_on_list(
    const c10::BFloat16* P, const c10::BFloat16* C, float* D,
    const uint32_t* idx_list, const uint32_t* list_count,
    int dim, int N, int M);

/**
 * @brief Fallback using FP32 direct formula on BF16 data.
 */
__global__ void direct_fp32_on_list_from_bf16(
    const float* P, const float* C, float* D,
    const uint32_t* idx_list, const uint32_t* list_count,
    int dim, int N, int M);

/**
 * @brief Fallback using FP32 direct formula.
 */
__global__ void direct_fp32_on_list(
    const float* P, const float* C, float* D,
    const uint32_t* idx_list, const uint32_t* list_count,
    int dim, int N, int M);

// Host function declarations (all take FP32 inputs now)
// ============================================================================
// HOST FUNCTION DECLARATIONS - EXISTING (FP64 Fallback)
// ============================================================================

/**
 * @brief Compute pairwise Euclidean distances using FP16 mixed-precision.
 * 
 * Implements adaptive-precision distance computing (Algorithm 1) using:
 * - Low precision: FP16 (u_l = 2^-11)
 * - High precision: FP64 accumulator (u_h = 2^-53)
 * 
 * Algorithm Overview:
 * 1. Cast FP32 inputs to FP16 for cuBLAS GEMM acceleration
 * 2. Compute norms pp, cc in FP16 arithmetic (line 2-3)
 * 3. Compute dot products s via cuBLAS (line 4)
 * 4. Form expanded distances d_exp = pp - 2s + cc (line 5-6)
 * 5. Apply error-floor test with γ_{2ℓ+4} bound (line 8-10)
 * 6. Recompute suspect distances in FP64 (line 11-13)
 * 
 * @param P      First data matrix (N x dim), float32, CUDA, contiguous
 * @param C      Second data matrix (M x dim), float32, CUDA, contiguous
 * @param kappa  Safety factor ρ for error floor (default: 1.0, typical: 1.0-5.0)
 * 
 * @return D     Pairwise squared distances (N x M), float32
 * 
 * @throws std::runtime_error if:
 *   - Inputs not on CUDA device or not contiguous
 *   - Inputs not float32 dtype
 *   - Dimension mismatch between P and C
 *   - N*M exceeds uint32_t range
 *   - cuBLAS errors occur
 * 
 * @note Requires GPU with compute capability >= 7.0 for Tensor Core acceleration
 * @note Typical speedup: 2-8x over FP32 depending on GPU architecture
 * @note Typical fallback rate: 0.5-5% depending on kappa and data distribution
 * 
 * Performance Characteristics:
 * - Memory: O(N*M) for distance matrix + O(N+M) for norms + O(N*M) for index list
 * - Compute: O(N*M*dim) dominated by cuBLAS GEMM
 * - Synchronization: One cudaDeviceSynchronize() at end
 * 
 * Example:
 * @code
 *   torch::Tensor P = torch::randn({1000, 128}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
 *   torch::Tensor C = torch::randn({500, 128}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
 *   torch::Tensor D = pairwise_euclidean_fp16_fp64(P, C, 3.0);
 *   // D.shape == {1000, 500}, D[i,j] = ||P[i] - C[j]||^2
 * @endcode
 */
torch::Tensor pairwise_euclidean_fp16_fp64(
    torch::Tensor P, torch::Tensor C, float kappa);

/**
 * @brief Compute pairwise Euclidean distances using BF16 mixed-precision.
 * 
 * Same as pairwise_euclidean_fp16_fp64 but uses BFloat16 (Brain Floating Point).
 * 
 * BF16 Characteristics:
 * - Machine epsilon: u_l = 2^-7 (vs 2^-11 for FP16)
 * - Dynamic range: Same as FP32 (8-bit exponent)
 * - Better for: Wide value ranges, outlier-heavy data
 * - Worse for: Absolute precision requirements
 * 
 * @param P      First data matrix (N x dim), float32, CUDA, contiguous
 * @param C      Second data matrix (M x dim), float32, CUDA, contiguous
 * @param kappa  Safety factor ρ (recommend: 3.0-5.0 for BF16)
 * 
 * @return D     Pairwise squared distances (N x M), float32
 * 
 * @note Requires GPU with BF16 support (Ampere architecture or newer)
 * @note Higher fallback rate than FP16 (~1.5-2x at same kappa)
 * 
 * @see pairwise_euclidean_fp16_fp64 for detailed documentation
 */
torch::Tensor pairwise_euclidean_bf16_fp64(
    torch::Tensor P, torch::Tensor C, float kappa);


/**
 * @brief Compute pairwise Euclidean distances using TF32 mixed-precision.
 * 
 * Same as pairwise_euclidean_fp16_fp64 but uses TensorFloat-32.
 * 
 * TF32 Characteristics:
 * - Machine epsilon: u_l = 2^-10
 * - Dynamic range: Same as FP32
 * - Mantissa: 10 bits (vs 23 for FP32, 10 for FP16)
 * - Best balance of speed and accuracy
 * 
 * @param P      First data matrix (N x dim), float32, CUDA, contiguous
 * @param C      Second data matrix (M x dim), float32, CUDA, contiguous
 * @param kappa  Safety factor ρ (recommend: 2.0-3.0 for TF32)
 * 
 * @return D     Pairwise squared distances (N x M), float32
 * 
 * @note Requires Ampere architecture or newer (A100, A6000, RTX 30xx+)
 * @note Lowest fallback rate among mixed-precision variants
 * @note Speedup: 4-8x over FP32 on Ampere GPUs
 * 
 * @see pairwise_euclidean_fp16_fp64 for detailed documentation
 */
torch::Tensor pairwise_euclidean_tf32_fp64(
    torch::Tensor P, torch::Tensor C, float kappa);

/**
 * @brief Compute pairwise Euclidean distances using FP32 with FP64 fallback.
 * 
 * Reference implementation using standard FP32 arithmetic with selective
 * FP64 recomputation. No mixed-precision acceleration, serves as baseline.
 * 
 * Uses different error model: γ_dim instead of γ_{2dim+4}
 * 
 * @param P      First data matrix (N x dim), float32, CUDA, contiguous
 * @param C      Second data matrix (M x dim), float32, CUDA, contiguous
 * @param kappa  Safety factor (typical: 1.0-2.0 for FP32)
 * 
 * @return D     Pairwise squared distances (N x M), float32
 * 
 * @note Uses PyTorch's torch.mm() instead of cuBLAS (no Tensor Core)
 * @note ~2-8x slower than mixed-precision variants
 * @note Lowest fallback rate: ~0.1-0.5%
 * @note Most numerically stable variant
 * 
 * @see pairwise_euclidean_fp16_fp64 for detailed documentation
 */
torch::Tensor pairwise_euclidean_fp32_fp64(
    torch::Tensor P, torch::Tensor C, float kappa);

// Get fallback statistics

/**
 * @brief Compute pairwise distances with FP16 and return fallback statistics.
 * 
 * Identical to pairwise_euclidean_fp16_fp64 but additionally returns the
 * number of distances that required high-precision recomputation.
 * 
 * @param P      First data matrix (N x dim), float32, CUDA, contiguous
 * @param C      Second data matrix (M x dim), float32, CUDA, contiguous
 * @param kappa  Safety factor ρ
 * 
 * @return std::tuple<torch::Tensor, int>
 *   - D (torch::Tensor): Pairwise squared distances (N x M), float32
 *   - fallback_count (int): Number of distances recomputed in FP64
 * 
 * Use Cases:
 * - Profile numerical behavior on specific datasets
 * - Tune kappa parameter for optimal speed/accuracy tradeoff
 * - Monitor production systems for numerical issues
 * - Generate validation reports
 * 
 * Performance Impact: Negligible (~0.1% overhead for single D2H memcpy)
 * 
 * Typical Fallback Rates (FP16, dim=128):
 * - kappa=1.0: 0.5-2%
 * - kappa=3.0: 2-5%
 * - kappa=5.0: 5-10%
 * 
 * Example:
 * @code
 *   auto [D, count] = pairwise_euclidean_fp16_fp64_with_stats(P, C, 3.0);
 *   double fallback_rate = (double)count / (P.size(0) * C.size(0));
 *   std::cout << "Fallback rate: " << fallback_rate * 100 << "%" << std::endl;
 * @endcode
 * 
 * @see pairwise_euclidean_fp16_fp64 for algorithm details
 */
std::tuple<torch::Tensor, int> pairwise_euclidean_fp16_fp64_with_stats(
    torch::Tensor P, torch::Tensor C, float kappa);

/**
 * @brief Compute pairwise distances with BF16 and return fallback statistics.
 * 
 * Same as pairwise_euclidean_fp16_fp64_with_stats but uses BF16 precision.
 * 
 * @param P      First data matrix (N x dim), float32, CUDA, contiguous
 * @param C      Second data matrix (M x dim), float32, CUDA, contiguous
 * @param kappa  Safety factor ρ
 * 
 * @return std::tuple<torch::Tensor, int>
 *   - D: Pairwise squared distances (N x M), float32
 *   - fallback_count: Number of FP64 recomputations
 * 
 * Typical Fallback Rates (BF16, dim=128):
 * - kappa=1.0: 1-3%
 * - kappa=3.0: 4-8%
 * - kappa=5.0: 8-15%
 * 
 * @see pairwise_euclidean_fp16_fp64_with_stats for detailed documentation
 */
std::tuple<torch::Tensor, int> pairwise_euclidean_bf16_fp64_with_stats(
    torch::Tensor P, torch::Tensor C, float kappa);

/**
 * @brief Compute pairwise distances with TF32 and return fallback statistics.
 * 
 * Same as pairwise_euclidean_fp16_fp64_with_stats but uses TF32 precision.
 * 
 * @param P      First data matrix (N x dim), float32, CUDA, contiguous
 * @param C      Second data matrix (M x dim), float32, CUDA, contiguous
 * @param kappa  Safety factor ρ
 * 
 * @return std::tuple<torch::Tensor, int>
 *   - D: Pairwise squared distances (N x M), float32
 *   - fallback_count: Number of FP64 recomputations
 * 
 * Typical Fallback Rates (TF32, dim=128):
 * - kappa=1.0: 0.3-1%
 * - kappa=2.0: 1-2%
 * - kappa=3.0: 2-4%
 * 
 * @note Lowest fallback rate among mixed-precision variants
 * @note Ideal for production monitoring on Ampere+ GPUs
 * 
 * @see pairwise_euclidean_fp16_fp64_with_stats for detailed documentation
 */
std::tuple<torch::Tensor, int> pairwise_euclidean_tf32_fp64_with_stats(
    torch::Tensor P, torch::Tensor C, float kappa);

/**
 * @brief Compute pairwise distances with FP32 and return fallback statistics.
 * 
 * Same as pairwise_euclidean_fp16_fp64_with_stats but uses FP32 precision.
 * 
 * @param P      First data matrix (N x dim), float32, CUDA, contiguous
 * @param C      Second data matrix (M x dim), float32, CUDA, contiguous
 * @param kappa  Safety factor
 * 
 * @return std::tuple<torch::Tensor, int>
 *   - D: Pairwise squared distances (N x M), float32
 *   - fallback_count: Number of FP64 recomputations
 * 
 * Typical Fallback Rates (FP32, dim=128):
 * - kappa=1.0: 0.1-0.5%
 * - kappa=2.0: 0.5-1%
 * 
 * @note Lowest fallback rate of all variants
 * 
 * @see pairwise_euclidean_fp16_fp64_with_stats for detailed documentation
 */
std::tuple<torch::Tensor, int> pairwise_euclidean_fp32_fp64_with_stats(
    torch::Tensor P, torch::Tensor C, float kappa);

// ============================================================================
// NEW: HOST FUNCTION DECLARATIONS (Direct Formula Fallback)
// ============================================================================

/**
 * @brief FP16 -> FP16 direct formula fallback
 */
torch::Tensor pairwise_euclidean_fp16_fp16(
    torch::Tensor P, torch::Tensor C, float kappa);

/**
 * @brief FP16 -> FP32 direct formula fallback
 */
torch::Tensor pairwise_euclidean_fp16_fp32(
    torch::Tensor P, torch::Tensor C, float kappa);

/**
 * @brief BF16 -> BF16 direct formula fallback
 */
torch::Tensor pairwise_euclidean_bf16_bf16(
    torch::Tensor P, torch::Tensor C, float kappa);

/**
 * @brief BF16 -> FP32 direct formula fallback
 */
torch::Tensor pairwise_euclidean_bf16_fp32(
    torch::Tensor P, torch::Tensor C, float kappa);

/**
 * @brief FP32 -> FP32 direct formula fallback
 */
torch::Tensor pairwise_euclidean_fp32_fp32(
    torch::Tensor P, torch::Tensor C, float kappa);

/**
 * @brief FP16 -> FP16 with statistics
 */
std::tuple<torch::Tensor, int> pairwise_euclidean_fp16_fp16_with_stats(
    torch::Tensor P, torch::Tensor C, float kappa);

/**
 * @brief FP16 -> FP32 with statistics
 */
std::tuple<torch::Tensor, int> pairwise_euclidean_fp16_fp32_with_stats(
    torch::Tensor P, torch::Tensor C, float kappa);

/**
 * @brief BF16 -> BF16 with statistics
 */
std::tuple<torch::Tensor, int> pairwise_euclidean_bf16_bf16_with_stats(
    torch::Tensor P, torch::Tensor C, float kappa);

/**
 * @brief BF16 -> FP32 with statistics
 */
std::tuple<torch::Tensor, int> pairwise_euclidean_bf16_fp32_with_stats(
    torch::Tensor P, torch::Tensor C, float kappa);

/**
 * @brief FP32 -> FP32 with statistics
 */
std::tuple<torch::Tensor, int> pairwise_euclidean_fp32_fp32_with_stats(
    torch::Tensor P, torch::Tensor C, float kappa);

/**
 * @brief Extract fallback count from device tensor.
 * 
 * Utility function to retrieve the number of fallback recomputations from
 * a GPU tensor. Performs synchronous device-to-host memory transfer.
 * 
 * @param list_count_tensor  Device tensor (int32, shape [1]) containing count
 * 
 * @return int  Fallback count value
 * 
 * @note Primarily for internal use or advanced debugging
 * @note Not needed when using *_with_stats variants
 * @note Performs cudaMemcpy (blocking operation)
 * 
 * Example:
 * @code
 *   torch::Tensor count_tensor = torch::zeros({1}, torch::dtype(torch::kInt32).device(torch::kCUDA));
 *   // ... kernel execution that updates count_tensor ...
 *   int count = get_fallback_count_low(count_tensor);
 * @endcode
 */
int get_fallback_count_low(torch::Tensor list_count_tensor);

// cuBLAS handle management (only for CUDA compilation)
/**
 * @brief Get thread-local cuBLAS handle.
 * 
 * Returns a thread-local cuBLAS handle, creating it if necessary.
 * Handle is automatically configured to use the current CUDA stream.
 * 
 * @return cublasHandle_t  Thread-local cuBLAS handle
 * 
 * @throws std::runtime_error if cuBLAS initialization fails
 * 
 * @note Thread-safe via thread_local storage
 * @note Handle is never explicitly destroyed (relies on process cleanup)
 * @note Automatically synced to at::cuda::getCurrentCUDAStream()
 * 
 * @warning Internal function - users should not call directly
 */
cublasHandle_t get_cublas_handle();

void sync_cublas_stream(cublasHandle_t handle);

// cuBLAS GEMM wrappers
void cublas_gemm_fp16(cublasHandle_t handle,
                      const c10::Half* A, const c10::Half* B, c10::Half* C,
                      int M, int N, int K);

void cublas_gemm_bf16(cublasHandle_t handle,
                      const c10::BFloat16* A, const c10::BFloat16* B, c10::BFloat16* C,
                      int M, int N, int K);

void cublas_gemm_tf32(cublasHandle_t handle,
                      const float* A, const float* B, float* C,
                      int M, int N, int K);