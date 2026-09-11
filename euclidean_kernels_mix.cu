/*
 * PyTorch CUDA Extension: Mixed-Precision with cuBLAS optimization
 *
 * ALGORITHM-COMPLIANT VERSION:
 * - All inputs are FP32 (working precision u).
 * - Fast path: Cast FP32 to low-precision, compute norms/dot in low-precision arithmetic (fl_ul).
 * - Error floor: γ_{2 ell + 4} formula.
 * - Fallback: High-precision (FP64 accumulator) direct distance on original FP32 data.
 */
#include "euclidean_kernels_mix.cuh"
#include "euclidean_kernels.cuh"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <ATen/cuda/CUDAContext.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t _cublas_status = (call); \
        if (_cublas_status != CUBLAS_STATUS_SUCCESS) { \
            throw std::runtime_error("cuBLAS error at " + std::string(__FILE__) + ":" + std::to_string(__LINE__)); \
        } \
    } while(0)

// ============================================================================
// CUBLAS HANDLE (thread-local for safety)
// ============================================================================

static thread_local cublasHandle_t g_cublas_handle = nullptr;

cublasHandle_t get_cublas_handle() {
    if (g_cublas_handle == nullptr) {
        CUBLAS_CHECK(cublasCreate(&g_cublas_handle));
    }
    return g_cublas_handle;
}

void sync_cublas_stream(cublasHandle_t handle) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    CUBLAS_CHECK(cublasSetStream(handle, stream));
}

// ============================================================================
// TYPE CONVERSION HELPERS
// ============================================================================

__device__ __forceinline__ __half to_cuda_half(const c10::Half& x) {
    return *reinterpret_cast<const __half*>(&x);
}

__device__ __forceinline__ __nv_bfloat16 to_cuda_bfloat16(const c10::BFloat16& x) {
    return *reinterpret_cast<const __nv_bfloat16*>(&x);
}

// ============================================================================
// NORM KERNELS (Algorithm-compliant: cast FP32 to low-prec, compute in low-prec)
// ============================================================================

__global__ void compute_norms_fp16(const float* X, float* X2, int N, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    const float* row = X + idx * dim;
    
    // fl_ul(p^T p): cast to FP16, accumulate in FP32 (FP16 arithmetic)
    float s = 0.f;
    for (int k = 0; k < dim; ++k) {
        __half h = __float2half(row[k]);  // Cast to low-precision
        float v = __half2float(h);         // Convert back for accumulation
        s = fmaf(v, v, s);                 // FP32 accumulation (hardware does FP16 multiply)
    }
    X2[idx] = s;
}

__global__ void compute_norms_bf16(const float* X, float* X2, int N, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    const float* row = X + idx * dim;
    
    float s = 0.f;
    for (int k = 0; k < dim; ++k) {
        __nv_bfloat16 bf = __float2bfloat16(row[k]);
        float v = __bfloat162float(bf);
        s = fmaf(v, v, s);
    }
    X2[idx] = s;
}

__global__ void compute_norms_tf32(const float* X, float* X2, int N, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    const float* row = X + idx * dim;
    
    // TF32: No explicit cast needed, cuBLAS handles it
    float s = 0.f;
    for (int k = 0; k < dim; ++k) {
        float v = row[k];
        s = fmaf(v, v, s);
    }
    X2[idx] = s;
}

__global__ void compute_norms_fp16_uniform(
    const c10::Half* X, c10::Half* X2, int N, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    const c10::Half* row = X + idx * dim;
    float s = 0.f;  // Use FP32 accumulator for stability
    for (int k = 0; k < dim; ++k) {
        __half h = to_cuda_half(row[k]);
        float v = __half2float(h);
        s = fmaf(v, v, s);
    }
    X2[idx] = __float2half_rn(s);
}


__global__ void compute_norms_bf16_uniform(
    const c10::BFloat16* X, c10::BFloat16* X2, int N, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    const c10::BFloat16* row = X + idx * dim;
    float s = 0.f;
    for (int k = 0; k < dim; ++k) {
        __nv_bfloat16 bf = to_cuda_bfloat16(row[k]);
        float v = __bfloat162float(bf);
        s = fmaf(v, v, s);
    }
    X2[idx] = __float2bfloat16_rn(s);
}


__global__ void compute_norms_tf32_uniform(
    const float* X, float* X2, int N, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    const float* row = X + idx * dim;
    float s = 0.f;
    for (int k = 0; k < dim; ++k) {
        float v = row[k];
        s = fmaf(v, v, s);
    }
    X2[idx] = s;
}

// ============================================================================
// FALLBACK RECOMPUTE (High-precision: FP64 accumulator on original FP32)
// ============================================================================

__global__ void direct_on_list_rowmajor(const float* P, const float* C, float* D,
                                        const uint32_t* idx_list, const uint32_t* list_count,
                                        int dim, int N, int M) {
    uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t cnt = *list_count;
    if (t >= cnt) return;
    uint32_t idx = idx_list[t];
    int i = idx / M;
    int j = idx % M;
    if (i >= N || j >= M) return;
    
    const float* p = P + i * dim;
    const float* c = C + j * dim;
    
    // fl_uh((p - c)^T(p - c)): FP64 accumulator
    double acc = 0.0;
    for (int k = 0; k < dim; ++k) {
        double diff = (double)p[k] - (double)c[k];
        acc += diff * diff;
    }
    D[(size_t)i * M + j] = (float)acc;
}

// ============================================================================
// DISTANCE FORMATION + ERROR-FLOOR TEST KERNELS
// Algorithm: ε_floor = ρ · γ_{2ℓ+4} · (pp + cc)
// ============================================================================

__global__ void form_mixed_and_list_fp16(
    const c10::Half* S, const c10::Half* P2, const c10::Half* C2,
    float* D, uint32_t* idx_list, uint32_t* list_count,
    int N, int M, int dim, float rho) {
    
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || j >= M) return;

    size_t pos = (size_t)i * M + j;
    
    // d_exp = fl_ul(pp - 2s + cc)
    __half s_h = to_cuda_half(S[pos]);
    float s = __half2float(s_h);
    __half pp_h = to_cuda_half(P2[i]);
    float pp = __half2float(pp_h);
    __half cc_h = to_cuda_half(C2[j]);
    float cc = __half2float(cc_h);
    
    float d_exp = pp - 2.f * s + cc;
    d_exp = fmaxf(d_exp, 0.f);
    D[pos] = d_exp;

    // Error-floor test: ε_floor = ρ · γ_{2ℓ+4} · (pp + cc)
    const float u_fp16 = 4.8828125e-4f;  // 2^-11
    int r = 2 * dim + 4;
    float gamma_r = (float)r * u_fp16;
    float eps_floor = rho * gamma_r * (pp + cc);

    if (d_exp <= eps_floor) {
        uint32_t p = atomicAdd(list_count, 1u);
        idx_list[p] = (uint32_t)pos;
    }
}

__global__ void form_mixed_and_list_bf16(
    const c10::BFloat16* S, const c10::BFloat16* P2, const c10::BFloat16* C2,
    float* D, uint32_t* idx_list, uint32_t* list_count,
    int N, int M, int dim, float rho) {
    
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || j >= M) return;

    size_t pos = (size_t)i * M + j;
    
    __nv_bfloat16 s_bf = to_cuda_bfloat16(S[pos]);
    float s = __bfloat162float(s_bf);
    __nv_bfloat16 pp_bf = to_cuda_bfloat16(P2[i]);
    float pp = __bfloat162float(pp_bf);
    __nv_bfloat16 cc_bf = to_cuda_bfloat16(C2[j]);
    float cc = __bfloat162float(cc_bf);
    
    float d_exp = pp - 2.f * s + cc;
    d_exp = fmaxf(d_exp, 0.f);
    D[pos] = d_exp;

    const float u_bf16 = 7.8125e-3f;  // 2^-7
    int r = 2 * dim + 4;
    float gamma_r = (float)r * u_bf16;
    float eps_floor = rho * gamma_r * (pp + cc);

    if (d_exp <= eps_floor) {
        uint32_t p = atomicAdd(list_count, 1u);
        idx_list[p] = (uint32_t)pos;
    }
}

__global__ void form_mixed_and_list_tf32(
    const float* S, const float* P2, const float* C2,
    float* D, uint32_t* idx_list, uint32_t* list_count,
    int N, int M, int dim, float rho) {
    
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || j >= M) return;

    size_t pos = (size_t)i * M + j;
    
    float s = S[pos];
    float pp = P2[i];
    float cc = C2[j];
    
    float d_exp = pp - 2.f * s + cc;
    d_exp = fmaxf(d_exp, 0.f);
    D[pos] = d_exp;

    const float u_tf32 = 4.8828125e-4f;  // 2^-11
    int r = 2 * dim + 4;
    float gamma_r = (float)r * u_tf32;
    float eps_floor = rho * gamma_r * (pp + cc);

    if (d_exp <= eps_floor) {
        uint32_t p = atomicAdd(list_count, 1u);
        idx_list[p] = (uint32_t)pos;
    }
}

// ============================================================================
// FP32 ERROR-FLOOR TEST KERNEL (separate from TF32)
// ============================================================================

__global__ void form_mixed_and_list_fp32(
    const float* S, const float* P2, const float* C2,
    float* D, uint32_t* idx_list, uint32_t* list_count,
    int N, int M, int dim, float rho) {
    
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || j >= M) return;

    size_t pos = (size_t)i * M + j;
    
    // d_exp = fl(pp - 2s + cc)
    float s = S[pos];
    float pp = P2[i];
    float cc = C2[j];
    
    float d_exp = pp - 2.f * s + cc;
    d_exp = fmaxf(d_exp, 0.f);
    D[pos] = d_exp;

    // Error-floor test: ε_floor = ρ · γ_{2ℓ+4} · (pp + cc)
    const float u_fp32 = 1.1920929e-7f;  // 2^-23 (FP32 machine epsilon)
    int r = 2 * dim + 4;
    float gamma_r = (float)r * u_fp32;
    float eps_floor = rho * gamma_r * (pp + cc);

    if (d_exp <= eps_floor) {
        uint32_t p = atomicAdd(list_count, 1u);
        idx_list[p] = (uint32_t)pos;
    }
}

// ============================================================================
// CUBLAS GEMM WRAPPERS
// ============================================================================

void cublas_gemm_fp16(cublasHandle_t handle,
                      const c10::Half* A, const c10::Half* B, c10::Half* C,
                      int M, int N, int K) {
    float alpha = 1.0f;
    float beta = 0.0f;

    int lda = K;
    int ldb = K;
    int ldc = N;

    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        N,
        M,
        K,
        &alpha,
        (const void*)B, CUDA_R_16F, ldb,
        (const void*)A, CUDA_R_16F, lda,
        &beta,
        (void*)C, CUDA_R_16F, ldc,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));
}

void cublas_gemm_bf16(cublasHandle_t handle,
                      const c10::BFloat16* A, const c10::BFloat16* B, c10::BFloat16* C,
                      int M, int N, int K) {
    float alpha = 1.0f;
    float beta = 0.0f;

    int lda = K;
    int ldb = K;
    int ldc = N;

    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        N,
        M,
        K,
        &alpha,
        (const void*)B, CUDA_R_16BF, ldb,
        (const void*)A, CUDA_R_16BF, lda,
        &beta,
        (void*)C, CUDA_R_16BF, ldc,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));
}

void cublas_gemm_tf32(cublasHandle_t handle,
                      const float* A, const float* B, float* C,
                      int M, int N, int K) {
    float alpha = 1.0f;
    float beta = 0.0f;

    int lda = K;
    int ldb = K;
    int ldc = N;

    cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);

    cublasStatus_t gemm_status = cublasSgemm(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        N,
        M,
        K,
        &alpha,
        B, ldb,
        A, lda,
        &beta,
        C, ldc
    );

    cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);

    if (gemm_status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS error at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
}




__global__ void direct_fp16_on_list(const c10::Half* P, const c10::Half* C, float* D,
                                     const uint32_t* idx_list, const uint32_t* list_count,
                                     int dim, int N, int M) {
    uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t cnt = *list_count;
    if (t >= cnt) return;
    uint32_t idx = idx_list[t];
    int i = idx / M;
    int j = idx % M;
    if (i >= N || j >= M) return;
    
    const c10::Half* p = P + i * dim;
    const c10::Half* c = C + j * dim;
    
    // Direct formula in FP16: (p - c)^T(p - c)
    __half acc = __float2half(0.0f);

    for (int k = 0; k < dim; ++k) {
        __half ph = to_cuda_half(p[k]);
        __half ch = to_cuda_half(c[k]);

        __half diff = __hsub(ph, ch);
        __half sq   = __hmul(diff, diff);
        acc = __hadd(acc, sq);
    }

    D[(size_t)i * M + j] = __half2float(acc);
}


__global__ void direct_bf16_on_list(const c10::BFloat16* P, const c10::BFloat16* C, float* D,
                                     const uint32_t* idx_list, const uint32_t* list_count,
                                     int dim, int N, int M) {
    uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t cnt = *list_count;
    if (t >= cnt) return;
    uint32_t idx = idx_list[t];
    int i = idx / M;
    int j = idx % M;
    if (i >= N || j >= M) return;
    
    const c10::BFloat16* p = P + i * dim;
    const c10::BFloat16* c = C + j * dim;
    
    // Direct formula in BF16: (p - c)^T(p - c)
    __nv_bfloat16 acc = __float2bfloat16(0.0f);

    for (int k = 0; k < dim; ++k) {
        __nv_bfloat16 pb = to_cuda_bfloat16(p[k]);
        __nv_bfloat16 cb = to_cuda_bfloat16(c[k]);

        __nv_bfloat16 diff = __hsub(pb, cb);
        __nv_bfloat16 sq   = __hmul(diff, diff);
        acc = __hadd(acc, sq);
    }

    D[(size_t)i * M + j] = __bfloat162float(acc);
}



// ============================================================================
// DIRECT FORMULA FALLBACK KERNELS - FP32
// ============================================================================





__global__ void direct_fp32_on_list(const float* P, const float* C, float* D,
                                     const uint32_t* idx_list, const uint32_t* list_count,
                                     int dim, int N, int M) {
    uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t cnt = *list_count;
    if (t >= cnt) return;
    uint32_t idx = idx_list[t];
    int i = idx / M;
    int j = idx % M;
    if (i >= N || j >= M) return;
    
    const float* p = P + i * dim;
    const float* c = C + j * dim;
    
    // Direct formula in FP32: (p - c)^T(p - c)
    float acc = 0.0f;
    for (int k = 0; k < dim; ++k) {
        float pf = p[k];
        float cf = c[k];
        float diff = pf - cf;
        acc += diff * diff;
    }
    D[(size_t)i * M + j] = acc;
}


// ============================================================================
// HOST IMPLEMENTATIONS (Algorithm-compliant)
// - Input: FP32 P, C
// - Fast path: Cast to low-prec for GEMM, compute norms in low-prec arithmetic
// - Error floor: γ_{2ℓ+4} formula
// - Fallback: FP64 accumulator on original FP32
// ============================================================================

torch::Tensor pairwise_euclidean_fp16_fp64(torch::Tensor P, torch::Tensor C, float kappa) {
    CHECK_INPUT(P);
    CHECK_INPUT(C);
    TORCH_CHECK(P.dtype() == torch::kFloat32, "P must be float32");
    TORCH_CHECK(C.dtype() == torch::kFloat32, "C must be float32");
    TORCH_CHECK(P.dim() == 2 && C.dim() == 2, "Inputs must be 2D");

    const int N = P.size(0);
    const int M = C.size(0);
    const int dim = P.size(1);
    TORCH_CHECK(C.size(1) == dim, "Dimension mismatch");

    const size_t total = (size_t)N * (size_t)M;
    TORCH_CHECK(total <= (size_t)UINT32_MAX,
        "N*M exceeds uint32 range; this kernel does not support such large problem sizes");

    // Cast FP32 to FP16 for GEMM
    auto P_low = P.to(torch::kFloat16);
    auto C_low = C.to(torch::kFloat16);

    auto P2 = torch::empty({N}, torch::TensorOptions().dtype(torch::kFloat16).device(P.device()));
    auto C2 = torch::empty({M}, torch::TensorOptions().dtype(torch::kFloat16).device(C.device()));
    auto D  = torch::empty({N, M}, torch::TensorOptions().dtype(torch::kFloat32).device(P.device()));
    auto S  = torch::empty({N, M}, torch::TensorOptions().dtype(torch::kFloat16).device(P.device()));

    // Compute norms: fl_ul(p^T p) - cast FP32 to FP16, compute in FP16 arithmetic
    const int threads = 256;
    compute_norms_fp16_uniform<<<(N + threads - 1) / threads, threads>>>(
        P_low.data_ptr<c10::Half>(), P2.data_ptr<c10::Half>(), N, dim);
    compute_norms_fp16_uniform<<<(M + threads - 1) / threads, threads>>>(
        C_low.data_ptr<c10::Half>(), C2.data_ptr<c10::Half>(), M, dim);

    // Compute dot products: fl_ul(P * C^T) using cuBLAS
    cublasHandle_t handle = get_cublas_handle();
    sync_cublas_stream(handle);

    cublas_gemm_fp16(handle,
                     P_low.data_ptr<c10::Half>(),
                     C_low.data_ptr<c10::Half>(),
                     S.data_ptr<c10::Half>(),
                     N, M, dim);

    auto idx_list = torch::empty({(int64_t)total},
        torch::TensorOptions().dtype(torch::kInt32).device(P.device()));
    auto list_count = torch::zeros({1},
        torch::TensorOptions().dtype(torch::kInt32).device(P.device()));

    // Form distances + error-floor test
    dim3 block(16, 16);
    dim3 grid((M + 15) / 16, (N + 15) / 16);
    form_mixed_and_list_fp16<<<grid, block>>>(
        S.data_ptr<c10::Half>(), P2.data_ptr<c10::Half>(), C2.data_ptr<c10::Half>(),
        D.data_ptr<float>(),
        (uint32_t*)idx_list.data_ptr<int32_t>(),
        (uint32_t*)list_count.data_ptr<int32_t>(),
        N, M, dim, kappa);

    // Fallback: High-precision recompute on original FP32
    const int blocks_list = ((int)total + threads - 1) / threads;
    direct_on_list_rowmajor<<<blocks_list, threads>>>(
        P.data_ptr<float>(), C.data_ptr<float>(), D.data_ptr<float>(),
        (uint32_t*)idx_list.data_ptr<int32_t>(),
        (uint32_t*)list_count.data_ptr<int32_t>(),
        dim, N, M);

    cudaDeviceSynchronize();
    return D;
}

torch::Tensor pairwise_euclidean_bf16_fp64(torch::Tensor P, torch::Tensor C, float kappa) {
    CHECK_INPUT(P);
    CHECK_INPUT(C);
    TORCH_CHECK(P.dtype() == torch::kFloat32, "P must be float32");
    TORCH_CHECK(C.dtype() == torch::kFloat32, "C must be float32");
    TORCH_CHECK(P.dim() == 2 && C.dim() == 2, "Inputs must be 2D");

    const int N = P.size(0);
    const int M = C.size(0);
    const int dim = P.size(1);
    TORCH_CHECK(C.size(1) == dim, "Dimension mismatch");

    const size_t total = (size_t)N * (size_t)M;
    TORCH_CHECK(total <= (size_t)UINT32_MAX,
        "N*M exceeds uint32 range; this kernel does not support such large problem sizes");

    auto P_low = P.to(torch::kBFloat16);
    auto C_low = C.to(torch::kBFloat16);

    auto P2 = torch::empty({N}, torch::TensorOptions().dtype(torch::kBFloat16).device(P.device()));
    auto C2 = torch::empty({M}, torch::TensorOptions().dtype(torch::kBFloat16).device(C.device()));
    auto D  = torch::empty({N, M}, torch::TensorOptions().dtype(torch::kFloat32).device(P.device()));
    auto S  = torch::empty({N, M}, torch::TensorOptions().dtype(torch::kBFloat16).device(P.device()));

    const int threads = 256;
    compute_norms_bf16_uniform<<<(N + threads - 1) / threads, threads>>>(
        P_low.data_ptr<c10::BFloat16>(), P2.data_ptr<c10::BFloat16>(), N, dim);
    compute_norms_bf16_uniform<<<(M + threads - 1) / threads, threads>>>(
        C_low.data_ptr<c10::BFloat16>(), C2.data_ptr<c10::BFloat16>(), M, dim);

    cublasHandle_t handle = get_cublas_handle();
    sync_cublas_stream(handle);

    cublas_gemm_bf16(handle,
                     P_low.data_ptr<c10::BFloat16>(),
                     C_low.data_ptr<c10::BFloat16>(),
                     S.data_ptr<c10::BFloat16>(),
                     N, M, dim);

    auto idx_list = torch::empty({(int64_t)total},
        torch::TensorOptions().dtype(torch::kInt32).device(P.device()));
    auto list_count = torch::zeros({1},
        torch::TensorOptions().dtype(torch::kInt32).device(P.device()));

    dim3 block(16, 16);
    dim3 grid((M + 15) / 16, (N + 15) / 16);
    form_mixed_and_list_bf16<<<grid, block>>>(
        S.data_ptr<c10::BFloat16>(), P2.data_ptr<c10::BFloat16>(), C2.data_ptr<c10::BFloat16>(),
        D.data_ptr<float>(),
        (uint32_t*)idx_list.data_ptr<int32_t>(),
        (uint32_t*)list_count.data_ptr<int32_t>(),
        N, M, dim, kappa);

    const int blocks_list = ((int)total + threads - 1) / threads;
    direct_on_list_rowmajor<<<blocks_list, threads>>>(
        P.data_ptr<float>(), C.data_ptr<float>(), D.data_ptr<float>(),
        (uint32_t*)idx_list.data_ptr<int32_t>(),
        (uint32_t*)list_count.data_ptr<int32_t>(),
        dim, N, M);

    cudaDeviceSynchronize();
    return D;
}

torch::Tensor pairwise_euclidean_tf32_fp64(torch::Tensor P, torch::Tensor C, float kappa) {
    CHECK_INPUT(P);
    CHECK_INPUT(C);
    TORCH_CHECK(P.dtype() == torch::kFloat32, "P must be float32");
    TORCH_CHECK(C.dtype() == torch::kFloat32, "C must be float32");
    TORCH_CHECK(P.dim() == 2 && C.dim() == 2, "Inputs must be 2D");

    const int N = P.size(0);
    const int M = C.size(0);
    const int dim = P.size(1);
    TORCH_CHECK(C.size(1) == dim, "Dimension mismatch");

    const size_t total = (size_t)N * (size_t)M;
    TORCH_CHECK(total <= (size_t)UINT32_MAX,
        "N*M exceeds uint32 range; this kernel does not support such large problem sizes");

    auto P2 = torch::empty({N}, P.options());
    auto C2 = torch::empty({M}, C.options());
    auto D  = torch::empty({N, M}, P.options());
    auto S  = torch::empty({N, M}, P.options());

    const int threads = 256;
    compute_norms_tf32_uniform<<<(N + threads - 1) / threads, threads>>>(
        P.data_ptr<float>(), P2.data_ptr<float>(), N, dim);
    compute_norms_tf32_uniform<<<(M + threads - 1) / threads, threads>>>(
        C.data_ptr<float>(), C2.data_ptr<float>(), M, dim);

    cublasHandle_t handle = get_cublas_handle();
    sync_cublas_stream(handle);

    cublas_gemm_tf32(handle,
                     P.data_ptr<float>(),
                     C.data_ptr<float>(),
                     S.data_ptr<float>(),
                     N, M, dim);

    auto idx_list = torch::empty({(int64_t)total},
        torch::TensorOptions().dtype(torch::kInt32).device(P.device()));
    auto list_count = torch::zeros({1},
        torch::TensorOptions().dtype(torch::kInt32).device(P.device()));

    dim3 block(16, 16);
    dim3 grid((M + 15) / 16, (N + 15) / 16);
    form_mixed_and_list_tf32<<<grid, block>>>(
        S.data_ptr<float>(), P2.data_ptr<float>(), C2.data_ptr<float>(),
        D.data_ptr<float>(),
        (uint32_t*)idx_list.data_ptr<int32_t>(),
        (uint32_t*)list_count.data_ptr<int32_t>(),
        N, M, dim, kappa);

    const int blocks_list = ((int)total + threads - 1) / threads;
    direct_on_list_rowmajor<<<blocks_list, threads>>>(
        P.data_ptr<float>(), C.data_ptr<float>(), D.data_ptr<float>(),
        (uint32_t*)idx_list.data_ptr<int32_t>(),
        (uint32_t*)list_count.data_ptr<int32_t>(),
        dim, N, M);

    cudaDeviceSynchronize();
    return D;
}

// ============================================================================
// WITH STATISTICS VERSIONS
// ============================================================================

std::tuple<torch::Tensor, int> pairwise_euclidean_fp16_fp64_with_stats(
    torch::Tensor P, torch::Tensor C, float kappa) {
    CHECK_INPUT(P);
    CHECK_INPUT(C);
    TORCH_CHECK(P.dtype() == torch::kFloat32, "P must be float32");
    TORCH_CHECK(C.dtype() == torch::kFloat32, "C must be float32");
    TORCH_CHECK(P.dim() == 2 && C.dim() == 2, "Inputs must be 2D");

    const int N = P.size(0);
    const int M = C.size(0);
    const int dim = P.size(1);
    TORCH_CHECK(C.size(1) == dim, "Dimension mismatch");

    const size_t total = (size_t)N * (size_t)M;
    TORCH_CHECK(total <= (size_t)UINT32_MAX,
        "N*M exceeds uint32 range; this kernel does not support such large problem sizes");

    auto P_low = P.to(torch::kFloat16);
    auto C_low = C.to(torch::kFloat16);

    auto P2 = torch::empty({N}, torch::TensorOptions().dtype(torch::kFloat16).device(P.device()));
    auto C2 = torch::empty({M}, torch::TensorOptions().dtype(torch::kFloat16).device(C.device()));
    auto D  = torch::empty({N, M}, torch::TensorOptions().dtype(torch::kFloat32).device(P.device()));
    auto S  = torch::empty({N, M}, torch::TensorOptions().dtype(torch::kFloat16).device(P.device()));

    const int threads = 256;
    compute_norms_fp16_uniform<<<(N + threads - 1) / threads, threads>>>(
        P_low.data_ptr<c10::Half>(), P2.data_ptr<c10::Half>(), N, dim);
    compute_norms_fp16_uniform<<<(M + threads - 1) / threads, threads>>>(
        C_low.data_ptr<c10::Half>(), C2.data_ptr<c10::Half>(), M, dim);

    cublasHandle_t handle = get_cublas_handle();
    sync_cublas_stream(handle);

    cublas_gemm_fp16(handle,
                     P_low.data_ptr<c10::Half>(),
                     C_low.data_ptr<c10::Half>(),
                     S.data_ptr<c10::Half>(),
                     N, M, dim);

    auto idx_list = torch::empty({(int64_t)total},
        torch::TensorOptions().dtype(torch::kInt32).device(P.device()));
    auto list_count = torch::zeros({1},
        torch::TensorOptions().dtype(torch::kInt32).device(P.device()));

    dim3 block(16, 16);
    dim3 grid((M + 15) / 16, (N + 15) / 16);
    form_mixed_and_list_fp16<<<grid, block>>>(
        S.data_ptr<c10::Half>(), P2.data_ptr<c10::Half>(), C2.data_ptr<c10::Half>(),
        D.data_ptr<float>(),
        (uint32_t*)idx_list.data_ptr<int32_t>(),
        (uint32_t*)list_count.data_ptr<int32_t>(),
        N, M, dim, kappa);

    const int blocks_list = ((int)total + threads - 1) / threads;
    direct_on_list_rowmajor<<<blocks_list, threads>>>(
        P.data_ptr<float>(), C.data_ptr<float>(), D.data_ptr<float>(),
        (uint32_t*)idx_list.data_ptr<int32_t>(),
        (uint32_t*)list_count.data_ptr<int32_t>(),
        dim, N, M);

    cudaDeviceSynchronize();

    int32_t count;
    cudaMemcpy(&count, list_count.data_ptr<int32_t>(), sizeof(int32_t), cudaMemcpyDeviceToHost);
    return std::make_tuple(D, (int)count);
}

std::tuple<torch::Tensor, int> pairwise_euclidean_bf16_fp64_with_stats(
    torch::Tensor P, torch::Tensor C, float kappa) {
    CHECK_INPUT(P);
    CHECK_INPUT(C);
    TORCH_CHECK(P.dtype() == torch::kFloat32, "P must be float32");
    TORCH_CHECK(C.dtype() == torch::kFloat32, "C must be float32");
    TORCH_CHECK(P.dim() == 2 && C.dim() == 2, "Inputs must be 2D");

    const int N = P.size(0);
    const int M = C.size(0);
    const int dim = P.size(1);
    TORCH_CHECK(C.size(1) == dim, "Dimension mismatch");

    const size_t total = (size_t)N * (size_t)M;
    TORCH_CHECK(total <= (size_t)UINT32_MAX,
        "N*M exceeds uint32 range; this kernel does not support such large problem sizes");

    auto P_low = P.to(torch::kBFloat16);
    auto C_low = C.to(torch::kBFloat16);

    auto P2 = torch::empty({N}, torch::TensorOptions().dtype(torch::kBFloat16).device(P.device()));
    auto C2 = torch::empty({M}, torch::TensorOptions().dtype(torch::kBFloat16).device(C.device()));
    auto D  = torch::empty({N, M}, torch::TensorOptions().dtype(torch::kFloat32).device(P.device()));
    auto S  = torch::empty({N, M}, torch::TensorOptions().dtype(torch::kBFloat16).device(P.device()));

    const int threads = 256;
    compute_norms_bf16_uniform<<<(N + threads - 1) / threads, threads>>>(
        P_low.data_ptr<c10::BFloat16>(), P2.data_ptr<c10::BFloat16>(), N, dim);
    compute_norms_bf16_uniform<<<(M + threads - 1) / threads, threads>>>(
        C_low.data_ptr<c10::BFloat16>(), C2.data_ptr<c10::BFloat16>(), M, dim);

    cublasHandle_t handle = get_cublas_handle();
    sync_cublas_stream(handle);

    cublas_gemm_bf16(handle,
                     P_low.data_ptr<c10::BFloat16>(),
                     C_low.data_ptr<c10::BFloat16>(),
                     S.data_ptr<c10::BFloat16>(),
                     N, M, dim);

    auto idx_list = torch::empty({(int64_t)total},
        torch::TensorOptions().dtype(torch::kInt32).device(P.device()));
    auto list_count = torch::zeros({1},
        torch::TensorOptions().dtype(torch::kInt32).device(P.device()));

    dim3 block(16, 16);
    dim3 grid((M + 15) / 16, (N + 15) / 16);
    form_mixed_and_list_bf16<<<grid, block>>>(
        S.data_ptr<c10::BFloat16>(), P2.data_ptr<c10::BFloat16>(), C2.data_ptr<c10::BFloat16>(),
        D.data_ptr<float>(),
        (uint32_t*)idx_list.data_ptr<int32_t>(),
        (uint32_t*)list_count.data_ptr<int32_t>(),
        N, M, dim, kappa);

    const int blocks_list = ((int)total + threads - 1) / threads;
    direct_on_list_rowmajor<<<blocks_list, threads>>>(
        P.data_ptr<float>(), C.data_ptr<float>(), D.data_ptr<float>(),
        (uint32_t*)idx_list.data_ptr<int32_t>(),
        (uint32_t*)list_count.data_ptr<int32_t>(),
        dim, N, M);

    cudaDeviceSynchronize();

    int32_t count;
    cudaMemcpy(&count, list_count.data_ptr<int32_t>(), sizeof(int32_t), cudaMemcpyDeviceToHost);
    return std::make_tuple(D, (int)count);
}

std::tuple<torch::Tensor, int> pairwise_euclidean_tf32_fp64_with_stats(
    torch::Tensor P, torch::Tensor C, float kappa) {
    CHECK_INPUT(P);
    CHECK_INPUT(C);
    TORCH_CHECK(P.dtype() == torch::kFloat32, "P must be float32");
    TORCH_CHECK(C.dtype() == torch::kFloat32, "C must be float32");
    TORCH_CHECK(P.dim() == 2 && C.dim() == 2, "Inputs must be 2D");

    const int N = P.size(0);
    const int M = C.size(0);
    const int dim = P.size(1);
    TORCH_CHECK(C.size(1) == dim, "Dimension mismatch");

    const size_t total = (size_t)N * (size_t)M;
    TORCH_CHECK(total <= (size_t)UINT32_MAX,
        "N*M exceeds uint32 range; this kernel does not support such large problem sizes");

    auto P2 = torch::empty({N}, P.options());
    auto C2 = torch::empty({M}, C.options());
    auto D  = torch::empty({N, M}, P.options());
    auto S  = torch::empty({N, M}, P.options());

    const int threads = 256;
    compute_norms_tf32_uniform<<<(N + threads - 1) / threads, threads>>>(
        P.data_ptr<float>(), P2.data_ptr<float>(), N, dim);
    compute_norms_tf32_uniform<<<(M + threads - 1) / threads, threads>>>(
        C.data_ptr<float>(), C2.data_ptr<float>(), M, dim);

    cublasHandle_t handle = get_cublas_handle();
    sync_cublas_stream(handle);

    cublas_gemm_tf32(handle,
                     P.data_ptr<float>(),
                     C.data_ptr<float>(),
                     S.data_ptr<float>(),
                     N, M, dim);

    auto idx_list = torch::empty({(int64_t)total},
        torch::TensorOptions().dtype(torch::kInt32).device(P.device()));
    auto list_count = torch::zeros({1},
        torch::TensorOptions().dtype(torch::kInt32).device(P.device()));

    dim3 block(16, 16);
    dim3 grid((M + 15) / 16, (N + 15) / 16);
    form_mixed_and_list_tf32<<<grid, block>>>(
        S.data_ptr<float>(), P2.data_ptr<float>(), C2.data_ptr<float>(),
        D.data_ptr<float>(),
        (uint32_t*)idx_list.data_ptr<int32_t>(),
        (uint32_t*)list_count.data_ptr<int32_t>(),
        N, M, dim, kappa);

    const int blocks_list = ((int)total + threads - 1) / threads;
    direct_on_list_rowmajor<<<blocks_list, threads>>>(
        P.data_ptr<float>(), C.data_ptr<float>(), D.data_ptr<float>(),
        (uint32_t*)idx_list.data_ptr<int32_t>(),
        (uint32_t*)list_count.data_ptr<int32_t>(),
        dim, N, M);

    cudaDeviceSynchronize();

    int32_t count;
    cudaMemcpy(&count, list_count.data_ptr<int32_t>(), sizeof(int32_t), cudaMemcpyDeviceToHost);
    return std::make_tuple(D, (int)count);
}

// ============================================================================
// FP32->FP64 FALLBACK REFERENCE (UNCHANGED)
// ============================================================================

torch::Tensor pairwise_euclidean_fp32_fp64(torch::Tensor P, torch::Tensor C, float kappa) {
    CHECK_INPUT(P);
    CHECK_INPUT(C);
    TORCH_CHECK(P.dtype() == torch::kFloat32, "P must be float32");
    TORCH_CHECK(C.dtype() == torch::kFloat32, "C must be float32");
    TORCH_CHECK(P.dim() == 2 && C.dim() == 2, "Inputs must be 2D");

    const int N = P.size(0);
    const int M = C.size(0);
    const int dim = P.size(1);
    TORCH_CHECK(C.size(1) == dim, "Dimension mismatch");

    const float u = 1.1920929e-7f;
    const float gamma_n = dim * u;

    auto P2 = torch::empty({N}, P.options());
    auto C2 = torch::empty({M}, C.options());
    auto D  = torch::empty({N, M}, P.options());

    const size_t total = (size_t)N * (size_t)M;
    auto idx_list = torch::empty({(int64_t)total},
        torch::TensorOptions().dtype(torch::kInt32).device(P.device()));
    auto list_count = torch::zeros({1},
        torch::TensorOptions().dtype(torch::kInt32).device(P.device()));

    const int threads = 256;
    compute_norms_rowmajor_float<<<(N + threads - 1) / threads, threads>>>(
        P.data_ptr<float>(), P2.data_ptr<float>(), N, dim);
    compute_norms_rowmajor_float<<<(M + threads - 1) / threads, threads>>>(
        C.data_ptr<float>(), C2.data_ptr<float>(), M, dim);

    auto S = torch::mm(P, C.t());

    dim3 block(16, 16);
    dim3 grid((M + 15) / 16, (N + 15) / 16);
    form_mixed_and_list_rowmajor<<<grid, block>>>(
        S.data_ptr<float>(), P2.data_ptr<float>(), C2.data_ptr<float>(),
        D.data_ptr<float>(),
        (uint32_t*)idx_list.data_ptr<int32_t>(),
        (uint32_t*)list_count.data_ptr<int32_t>(),
        N, M, gamma_n, kappa);

    const int blocks_list = ((int)total + threads - 1) / threads;
    direct_on_list_rowmajor<<<blocks_list, threads>>>(
        P.data_ptr<float>(), C.data_ptr<float>(), D.data_ptr<float>(),
        (uint32_t*)idx_list.data_ptr<int32_t>(),
        (uint32_t*)list_count.data_ptr<int32_t>(),
        dim, N, M);

    cudaDeviceSynchronize();
    return D;
}

std::tuple<torch::Tensor, int> pairwise_euclidean_fp32_fp64_with_stats(
    torch::Tensor P, torch::Tensor C, float kappa) {
    CHECK_INPUT(P);
    CHECK_INPUT(C);
    TORCH_CHECK(P.dtype() == torch::kFloat32, "P must be float32");
    TORCH_CHECK(C.dtype() == torch::kFloat32, "C must be float32");
    TORCH_CHECK(P.dim() == 2 && C.dim() == 2, "Inputs must be 2D");

    const int N = P.size(0);
    const int M = C.size(0);
    const int dim = P.size(1);
    TORCH_CHECK(C.size(1) == dim, "Dimension mismatch");

    const float u = 1.1920929e-7f;
    const float gamma_n = dim * u;

    auto P2 = torch::empty({N}, P.options());
    auto C2 = torch::empty({M}, C.options());
    auto D  = torch::empty({N, M}, P.options());

    const size_t total = (size_t)N * (size_t)M;
    auto idx_list = torch::empty({(int64_t)total},
        torch::TensorOptions().dtype(torch::kInt32).device(P.device()));
    auto list_count = torch::zeros({1},
        torch::TensorOptions().dtype(torch::kInt32).device(P.device()));

    const int threads = 256;
    compute_norms_rowmajor_float<<<(N + threads - 1) / threads, threads>>>(
        P.data_ptr<float>(), P2.data_ptr<float>(), N, dim);
    compute_norms_rowmajor_float<<<(M + threads - 1) / threads, threads>>>(
        C.data_ptr<float>(), C2.data_ptr<float>(), M, dim);

    auto S = torch::mm(P, C.t());

    dim3 block(16, 16);
    dim3 grid((M + 15) / 16, (N + 15) / 16);
    form_mixed_and_list_rowmajor<<<grid, block>>>(
        S.data_ptr<float>(), P2.data_ptr<float>(), C2.data_ptr<float>(),
        D.data_ptr<float>(),
        (uint32_t*)idx_list.data_ptr<int32_t>(),
        (uint32_t*)list_count.data_ptr<int32_t>(),
        N, M, gamma_n, kappa);

    const int blocks_list = ((int)total + threads - 1) / threads;
    direct_on_list_rowmajor<<<blocks_list, threads>>>(
        P.data_ptr<float>(), C.data_ptr<float>(), D.data_ptr<float>(),
        (uint32_t*)idx_list.data_ptr<int32_t>(),
        (uint32_t*)list_count.data_ptr<int32_t>(),
        dim, N, M);

    cudaDeviceSynchronize();

    int32_t count;
    cudaMemcpy(&count, list_count.data_ptr<int32_t>(), sizeof(int32_t), cudaMemcpyDeviceToHost);
    return std::make_tuple(D, (int)count);
}

int get_fallback_count_low(torch::Tensor list_count_tensor) {
    uint32_t count;
    cudaMemcpy(&count, list_count_tensor.data_ptr<int32_t>(),
               sizeof(uint32_t), cudaMemcpyDeviceToHost);
    return (int)count;
}



// ============================================================================
// FP16 -> FP16 DIRECT FALLBACK
// ============================================================================

torch::Tensor pairwise_euclidean_fp16_fp16(torch::Tensor P, torch::Tensor C, float kappa) {
    CHECK_INPUT(P);
    CHECK_INPUT(C);
    TORCH_CHECK(P.dtype() == torch::kFloat32, "P must be float32");
    TORCH_CHECK(C.dtype() == torch::kFloat32, "C must be float32");
    TORCH_CHECK(P.dim() == 2 && C.dim() == 2, "Inputs must be 2D");

    const int N = P.size(0);
    const int M = C.size(0);
    const int dim = P.size(1);
    TORCH_CHECK(C.size(1) == dim, "Dimension mismatch");

    const size_t total = (size_t)N * (size_t)M;
    TORCH_CHECK(total <= (size_t)UINT32_MAX, "N*M exceeds uint32 range");

    // Cast to FP16
    auto P_low = P.to(torch::kFloat16);
    auto C_low = C.to(torch::kFloat16);

    auto P2 = torch::empty({N}, torch::TensorOptions().dtype(torch::kFloat16).device(P.device()));
    auto C2 = torch::empty({M}, torch::TensorOptions().dtype(torch::kFloat16).device(C.device()));
    auto D  = torch::empty({N, M}, torch::TensorOptions().dtype(torch::kFloat32).device(P.device()));
    auto S  = torch::empty({N, M}, torch::TensorOptions().dtype(torch::kFloat16).device(P.device()));

    const int threads = 256;
    compute_norms_fp16_uniform<<<(N + threads - 1) / threads, threads>>>(
        P_low.data_ptr<c10::Half>(), P2.data_ptr<c10::Half>(), N, dim);
    compute_norms_fp16_uniform<<<(M + threads - 1) / threads, threads>>>(
        C_low.data_ptr<c10::Half>(), C2.data_ptr<c10::Half>(), M, dim);

    cublasHandle_t handle = get_cublas_handle();
    sync_cublas_stream(handle);
    cublas_gemm_fp16(handle, P_low.data_ptr<c10::Half>(), C_low.data_ptr<c10::Half>(),
                     S.data_ptr<c10::Half>(), N, M, dim);

    auto idx_list = torch::empty({(int64_t)total},
        torch::TensorOptions().dtype(torch::kInt32).device(P.device()));
    auto list_count = torch::zeros({1},
        torch::TensorOptions().dtype(torch::kInt32).device(P.device()));

    dim3 block(16, 16);
    dim3 grid((M + 15) / 16, (N + 15) / 16);
    form_mixed_and_list_fp16<<<grid, block>>>(
        S.data_ptr<c10::Half>(), P2.data_ptr<c10::Half>(), C2.data_ptr<c10::Half>(),
        D.data_ptr<float>(), (uint32_t*)idx_list.data_ptr<int32_t>(),
        (uint32_t*)list_count.data_ptr<int32_t>(), N, M, dim, kappa);

    // FALLBACK: FP16 direct formula
    const int blocks_list = ((int)total + threads - 1) / threads;
    direct_fp16_on_list<<<blocks_list, threads>>>(
        P_low.data_ptr<c10::Half>(), C_low.data_ptr<c10::Half>(), D.data_ptr<float>(),
        (uint32_t*)idx_list.data_ptr<int32_t>(),
        (uint32_t*)list_count.data_ptr<int32_t>(), dim, N, M);

    cudaDeviceSynchronize();
    return D;
}

std::tuple<torch::Tensor, int> pairwise_euclidean_fp16_fp16_with_stats(
    torch::Tensor P, torch::Tensor C, float kappa) {
    CHECK_INPUT(P);
    CHECK_INPUT(C);
    TORCH_CHECK(P.dtype() == torch::kFloat32, "P must be float32");
    TORCH_CHECK(C.dtype() == torch::kFloat32, "C must be float32");
    TORCH_CHECK(P.dim() == 2 && C.dim() == 2, "Inputs must be 2D");

    const int N = P.size(0);
    const int M = C.size(0);
    const int dim = P.size(1);
    TORCH_CHECK(C.size(1) == dim, "Dimension mismatch");

    const size_t total = (size_t)N * (size_t)M;
    TORCH_CHECK(total <= (size_t)UINT32_MAX, "N*M exceeds uint32 range");

    auto P_low = P.to(torch::kFloat16);
    auto C_low = C.to(torch::kFloat16);

    auto P2 = torch::empty({N}, torch::TensorOptions().dtype(torch::kFloat16).device(P.device()));
    auto C2 = torch::empty({M}, torch::TensorOptions().dtype(torch::kFloat16).device(C.device()));
    auto D  = torch::empty({N, M}, torch::TensorOptions().dtype(torch::kFloat32).device(P.device()));
    auto S  = torch::empty({N, M}, torch::TensorOptions().dtype(torch::kFloat16).device(P.device()));

    const int threads = 256;
    compute_norms_fp16_uniform<<<(N + threads - 1) / threads, threads>>>(
        P_low.data_ptr<c10::Half>(), P2.data_ptr<c10::Half>(), N, dim);
    compute_norms_fp16_uniform<<<(M + threads - 1) / threads, threads>>>(
        C_low.data_ptr<c10::Half>(), C2.data_ptr<c10::Half>(), M, dim);

    cublasHandle_t handle = get_cublas_handle();
    sync_cublas_stream(handle);
    cublas_gemm_fp16(handle, P_low.data_ptr<c10::Half>(), C_low.data_ptr<c10::Half>(),
                     S.data_ptr<c10::Half>(), N, M, dim);

    auto idx_list = torch::empty({(int64_t)total},
        torch::TensorOptions().dtype(torch::kInt32).device(P.device()));
    auto list_count = torch::zeros({1},
        torch::TensorOptions().dtype(torch::kInt32).device(P.device()));

    dim3 block(16, 16);
    dim3 grid((M + 15) / 16, (N + 15) / 16);
    form_mixed_and_list_fp16<<<grid, block>>>(
        S.data_ptr<c10::Half>(), P2.data_ptr<c10::Half>(), C2.data_ptr<c10::Half>(),
        D.data_ptr<float>(), (uint32_t*)idx_list.data_ptr<int32_t>(),
        (uint32_t*)list_count.data_ptr<int32_t>(), N, M, dim, kappa);

    const int blocks_list = ((int)total + threads - 1) / threads;
    direct_fp16_on_list<<<blocks_list, threads>>>(
        P_low.data_ptr<c10::Half>(), C_low.data_ptr<c10::Half>(), D.data_ptr<float>(),
        (uint32_t*)idx_list.data_ptr<int32_t>(),
        (uint32_t*)list_count.data_ptr<int32_t>(), dim, N, M);

    cudaDeviceSynchronize();

    int32_t count;
    cudaMemcpy(&count, list_count.data_ptr<int32_t>(), sizeof(int32_t), cudaMemcpyDeviceToHost);
    return std::make_tuple(D, (int)count);
}

// ============================================================================
// FP16 -> FP32 DIRECT FALLBACK
// ============================================================================

torch::Tensor pairwise_euclidean_fp16_fp32(torch::Tensor P, torch::Tensor C, float kappa) {
    CHECK_INPUT(P);
    CHECK_INPUT(C);
    TORCH_CHECK(P.dtype() == torch::kFloat32, "P must be float32");
    TORCH_CHECK(C.dtype() == torch::kFloat32, "C must be float32");
    TORCH_CHECK(P.dim() == 2 && C.dim() == 2, "Inputs must be 2D");

    const int N = P.size(0);
    const int M = C.size(0);
    const int dim = P.size(1);
    TORCH_CHECK(C.size(1) == dim, "Dimension mismatch");

    const size_t total = (size_t)N * (size_t)M;
    TORCH_CHECK(total <= (size_t)UINT32_MAX, "N*M exceeds uint32 range");

    auto P_low = P.to(torch::kFloat16);
    auto C_low = C.to(torch::kFloat16);

    auto P2 = torch::empty({N}, torch::TensorOptions().dtype(torch::kFloat16).device(P.device()));
    auto C2 = torch::empty({M}, torch::TensorOptions().dtype(torch::kFloat16).device(C.device()));
    auto D  = torch::empty({N, M}, torch::TensorOptions().dtype(torch::kFloat32).device(P.device()));
    auto S  = torch::empty({N, M}, torch::TensorOptions().dtype(torch::kFloat16).device(P.device()));

    const int threads = 256;
    compute_norms_fp16_uniform<<<(N + threads - 1) / threads, threads>>>(
        P_low.data_ptr<c10::Half>(), P2.data_ptr<c10::Half>(), N, dim);
    compute_norms_fp16_uniform<<<(M + threads - 1) / threads, threads>>>(
        C_low.data_ptr<c10::Half>(), C2.data_ptr<c10::Half>(), M, dim);

    cublasHandle_t handle = get_cublas_handle();
    sync_cublas_stream(handle);
    cublas_gemm_fp16(handle, P_low.data_ptr<c10::Half>(), C_low.data_ptr<c10::Half>(),
                     S.data_ptr<c10::Half>(), N, M, dim);

    auto idx_list = torch::empty({(int64_t)total},
        torch::TensorOptions().dtype(torch::kInt32).device(P.device()));
    auto list_count = torch::zeros({1},
        torch::TensorOptions().dtype(torch::kInt32).device(P.device()));

    dim3 block(16, 16);
    dim3 grid((M + 15) / 16, (N + 15) / 16);
    form_mixed_and_list_fp16<<<grid, block>>>(
        S.data_ptr<c10::Half>(), P2.data_ptr<c10::Half>(), C2.data_ptr<c10::Half>(),
        D.data_ptr<float>(), (uint32_t*)idx_list.data_ptr<int32_t>(),
        (uint32_t*)list_count.data_ptr<int32_t>(), N, M, dim, kappa);

    // FALLBACK: FP32 direct formula
    const int blocks_list = ((int)total + threads - 1) / threads;
    direct_fp32_on_list<<<blocks_list, threads>>>(
        P.data_ptr<float>(), C.data_ptr<float>(), D.data_ptr<float>(),
        (uint32_t*)idx_list.data_ptr<int32_t>(),
        (uint32_t*)list_count.data_ptr<int32_t>(), dim, N, M);

    cudaDeviceSynchronize();
    return D;
}

std::tuple<torch::Tensor, int> pairwise_euclidean_fp16_fp32_with_stats(
    torch::Tensor P, torch::Tensor C, float kappa) {
    CHECK_INPUT(P);
    CHECK_INPUT(C);
    TORCH_CHECK(P.dtype() == torch::kFloat32, "P must be float32");
    TORCH_CHECK(C.dtype() == torch::kFloat32, "C must be float32");
    TORCH_CHECK(P.dim() == 2 && C.dim() == 2, "Inputs must be 2D");

    const int N = P.size(0);
    const int M = C.size(0);
    const int dim = P.size(1);
    TORCH_CHECK(C.size(1) == dim, "Dimension mismatch");

    const size_t total = (size_t)N * (size_t)M;
    TORCH_CHECK(total <= (size_t)UINT32_MAX, "N*M exceeds uint32 range");

    auto P_low = P.to(torch::kFloat16);
    auto C_low = C.to(torch::kFloat16);

    auto P2 = torch::empty({N}, torch::TensorOptions().dtype(torch::kFloat16).device(P.device()));
    auto C2 = torch::empty({M}, torch::TensorOptions().dtype(torch::kFloat16).device(C.device()));
    auto D  = torch::empty({N, M}, torch::TensorOptions().dtype(torch::kFloat32).device(P.device()));
    auto S  = torch::empty({N, M}, torch::TensorOptions().dtype(torch::kFloat16).device(P.device()));

    const int threads = 256;
    compute_norms_fp16_uniform<<<(N + threads - 1) / threads, threads>>>(
        P_low.data_ptr<c10::Half>(), P2.data_ptr<c10::Half>(), N, dim);
    compute_norms_fp16_uniform<<<(M + threads - 1) / threads, threads>>>(
        C_low.data_ptr<c10::Half>(), C2.data_ptr<c10::Half>(), M, dim);

    cublasHandle_t handle = get_cublas_handle();
    sync_cublas_stream(handle);
    cublas_gemm_fp16(handle, P_low.data_ptr<c10::Half>(), C_low.data_ptr<c10::Half>(),
                     S.data_ptr<c10::Half>(), N, M, dim);

    auto idx_list = torch::empty({(int64_t)total},
        torch::TensorOptions().dtype(torch::kInt32).device(P.device()));
    auto list_count = torch::zeros({1},
        torch::TensorOptions().dtype(torch::kInt32).device(P.device()));

    dim3 block(16, 16);
    dim3 grid((M + 15) / 16, (N + 15) / 16);
    form_mixed_and_list_fp16<<<grid, block>>>(
        S.data_ptr<c10::Half>(), P2.data_ptr<c10::Half>(), C2.data_ptr<c10::Half>(),
        D.data_ptr<float>(), (uint32_t*)idx_list.data_ptr<int32_t>(),
        (uint32_t*)list_count.data_ptr<int32_t>(), N, M, dim, kappa);

    const int blocks_list = ((int)total + threads - 1) / threads;
    direct_fp32_on_list<<<blocks_list, threads>>>(
        P.data_ptr<float>(), C.data_ptr<float>(), D.data_ptr<float>(),
        (uint32_t*)idx_list.data_ptr<int32_t>(),
        (uint32_t*)list_count.data_ptr<int32_t>(), dim, N, M);

    cudaDeviceSynchronize();

    int32_t count;
    cudaMemcpy(&count, list_count.data_ptr<int32_t>(), sizeof(int32_t), cudaMemcpyDeviceToHost);
    return std::make_tuple(D, (int)count);
}

// ============================================================================
// BF16 -> BF16 DIRECT FALLBACK
// ============================================================================

torch::Tensor pairwise_euclidean_bf16_bf16(torch::Tensor P, torch::Tensor C, float kappa) {
    CHECK_INPUT(P);
    CHECK_INPUT(C);
    TORCH_CHECK(P.dtype() == torch::kFloat32, "P must be float32");
    TORCH_CHECK(C.dtype() == torch::kFloat32, "C must be float32");
    TORCH_CHECK(P.dim() == 2 && C.dim() == 2, "Inputs must be 2D");

    const int N = P.size(0);
    const int M = C.size(0);
    const int dim = P.size(1);
    TORCH_CHECK(C.size(1) == dim, "Dimension mismatch");

    const size_t total = (size_t)N * (size_t)M;
    TORCH_CHECK(total <= (size_t)UINT32_MAX, "N*M exceeds uint32 range");

    auto P_low = P.to(torch::kBFloat16);
    auto C_low = C.to(torch::kBFloat16);

    auto P2 = torch::empty({N}, torch::TensorOptions().dtype(torch::kBFloat16).device(P.device()));
    auto C2 = torch::empty({M}, torch::TensorOptions().dtype(torch::kBFloat16).device(C.device()));
    auto D  = torch::empty({N, M}, torch::TensorOptions().dtype(torch::kFloat32).device(P.device()));
    auto S  = torch::empty({N, M}, torch::TensorOptions().dtype(torch::kBFloat16).device(P.device()));

    const int threads = 256;
    compute_norms_bf16_uniform<<<(N + threads - 1) / threads, threads>>>(
        P_low.data_ptr<c10::BFloat16>(), P2.data_ptr<c10::BFloat16>(), N, dim);
    compute_norms_bf16_uniform<<<(M + threads - 1) / threads, threads>>>(
        C_low.data_ptr<c10::BFloat16>(), C2.data_ptr<c10::BFloat16>(), M, dim);

    cublasHandle_t handle = get_cublas_handle();
    sync_cublas_stream(handle);
    cublas_gemm_bf16(handle, P_low.data_ptr<c10::BFloat16>(), C_low.data_ptr<c10::BFloat16>(),
                     S.data_ptr<c10::BFloat16>(), N, M, dim);

    auto idx_list = torch::empty({(int64_t)total},
        torch::TensorOptions().dtype(torch::kInt32).device(P.device()));
    auto list_count = torch::zeros({1},
        torch::TensorOptions().dtype(torch::kInt32).device(P.device()));

    dim3 block(16, 16);
    dim3 grid((M + 15) / 16, (N + 15) / 16);
    form_mixed_and_list_bf16<<<grid, block>>>(
        S.data_ptr<c10::BFloat16>(), P2.data_ptr<c10::BFloat16>(), C2.data_ptr<c10::BFloat16>(),
        D.data_ptr<float>(), (uint32_t*)idx_list.data_ptr<int32_t>(),
        (uint32_t*)list_count.data_ptr<int32_t>(), N, M, dim, kappa);

    // FALLBACK: BF16 direct formula
    const int blocks_list = ((int)total + threads - 1) / threads;
    direct_bf16_on_list<<<blocks_list, threads>>>(
        P_low.data_ptr<c10::BFloat16>(), C_low.data_ptr<c10::BFloat16>(), D.data_ptr<float>(),
        (uint32_t*)idx_list.data_ptr<int32_t>(),
        (uint32_t*)list_count.data_ptr<int32_t>(), dim, N, M);

    cudaDeviceSynchronize();
    return D;
}

std::tuple<torch::Tensor, int> pairwise_euclidean_bf16_bf16_with_stats(
    torch::Tensor P, torch::Tensor C, float kappa) {
    CHECK_INPUT(P);
    CHECK_INPUT(C);
    TORCH_CHECK(P.dtype() == torch::kFloat32, "P must be float32");
    TORCH_CHECK(C.dtype() == torch::kFloat32, "C must be float32");
    TORCH_CHECK(P.dim() == 2 && C.dim() == 2, "Inputs must be 2D");

    const int N = P.size(0);
    const int M = C.size(0);
    const int dim = P.size(1);
    TORCH_CHECK(C.size(1) == dim, "Dimension mismatch");

    const size_t total = (size_t)N * (size_t)M;
    TORCH_CHECK(total <= (size_t)UINT32_MAX, "N*M exceeds uint32 range");

    auto P_low = P.to(torch::kBFloat16);
    auto C_low = C.to(torch::kBFloat16);

    auto P2 = torch::empty({N}, torch::TensorOptions().dtype(torch::kBFloat16).device(P.device()));
    auto C2 = torch::empty({M}, torch::TensorOptions().dtype(torch::kBFloat16).device(C.device()));
    auto D  = torch::empty({N, M}, torch::TensorOptions().dtype(torch::kFloat32).device(P.device()));
    auto S  = torch::empty({N, M}, torch::TensorOptions().dtype(torch::kBFloat16).device(P.device()));

    const int threads = 256;
    compute_norms_bf16_uniform<<<(N + threads - 1) / threads, threads>>>(
        P_low.data_ptr<c10::BFloat16>(), P2.data_ptr<c10::BFloat16>(), N, dim);
    compute_norms_bf16_uniform<<<(M + threads - 1) / threads, threads>>>(
        C_low.data_ptr<c10::BFloat16>(), C2.data_ptr<c10::BFloat16>(), M, dim);

    cublasHandle_t handle = get_cublas_handle();
    sync_cublas_stream(handle);
    cublas_gemm_bf16(handle, P_low.data_ptr<c10::BFloat16>(), C_low.data_ptr<c10::BFloat16>(),
                     S.data_ptr<c10::BFloat16>(), N, M, dim);

    auto idx_list = torch::empty({(int64_t)total},
        torch::TensorOptions().dtype(torch::kInt32).device(P.device()));
    auto list_count = torch::zeros({1},
        torch::TensorOptions().dtype(torch::kInt32).device(P.device()));

    dim3 block(16, 16);
    dim3 grid((M + 15) / 16, (N + 15) / 16);
    form_mixed_and_list_bf16<<<grid, block>>>(
        S.data_ptr<c10::BFloat16>(), P2.data_ptr<c10::BFloat16>(), C2.data_ptr<c10::BFloat16>(),
        D.data_ptr<float>(), (uint32_t*)idx_list.data_ptr<int32_t>(),
        (uint32_t*)list_count.data_ptr<int32_t>(), N, M, dim, kappa);

    const int blocks_list = ((int)total + threads - 1) / threads;
    direct_bf16_on_list<<<blocks_list, threads>>>(
        P_low.data_ptr<c10::BFloat16>(), C_low.data_ptr<c10::BFloat16>(), D.data_ptr<float>(),
        (uint32_t*)idx_list.data_ptr<int32_t>(),
        (uint32_t*)list_count.data_ptr<int32_t>(), dim, N, M);

    cudaDeviceSynchronize();

    int32_t count;
    cudaMemcpy(&count, list_count.data_ptr<int32_t>(), sizeof(int32_t), cudaMemcpyDeviceToHost);
    return std::make_tuple(D, (int)count);
}

// ============================================================================
// BF16 -> FP32 DIRECT FALLBACK
// ============================================================================

torch::Tensor pairwise_euclidean_bf16_fp32(torch::Tensor P, torch::Tensor C, float kappa) {
    CHECK_INPUT(P);
    CHECK_INPUT(C);
    TORCH_CHECK(P.dtype() == torch::kFloat32, "P must be float32");
    TORCH_CHECK(C.dtype() == torch::kFloat32, "C must be float32");
    TORCH_CHECK(P.dim() == 2 && C.dim() == 2, "Inputs must be 2D");

    const int N = P.size(0);
    const int M = C.size(0);
    const int dim = P.size(1);
    TORCH_CHECK(C.size(1) == dim, "Dimension mismatch");

    const size_t total = (size_t)N * (size_t)M;
    TORCH_CHECK(total <= (size_t)UINT32_MAX, "N*M exceeds uint32 range");

    auto P_low = P.to(torch::kBFloat16);
    auto C_low = C.to(torch::kBFloat16);

    auto P2 = torch::empty({N}, torch::TensorOptions().dtype(torch::kBFloat16).device(P.device()));
    auto C2 = torch::empty({M}, torch::TensorOptions().dtype(torch::kBFloat16).device(C.device()));
    auto D  = torch::empty({N, M}, torch::TensorOptions().dtype(torch::kFloat32).device(P.device()));
    auto S  = torch::empty({N, M}, torch::TensorOptions().dtype(torch::kBFloat16).device(P.device()));

    const int threads = 256;
    compute_norms_bf16_uniform<<<(N + threads - 1) / threads, threads>>>(
        P_low.data_ptr<c10::BFloat16>(), P2.data_ptr<c10::BFloat16>(), N, dim);
    compute_norms_bf16_uniform<<<(M + threads - 1) / threads, threads>>>(
        C_low.data_ptr<c10::BFloat16>(), C2.data_ptr<c10::BFloat16>(), M, dim);

    cublasHandle_t handle = get_cublas_handle();
    sync_cublas_stream(handle);
    cublas_gemm_bf16(handle, P_low.data_ptr<c10::BFloat16>(), C_low.data_ptr<c10::BFloat16>(),
                     S.data_ptr<c10::BFloat16>(), N, M, dim);

    auto idx_list = torch::empty({(int64_t)total},
        torch::TensorOptions().dtype(torch::kInt32).device(P.device()));
    auto list_count = torch::zeros({1},
        torch::TensorOptions().dtype(torch::kInt32).device(P.device()));

    dim3 block(16, 16);
    dim3 grid((M + 15) / 16, (N + 15) / 16);
    form_mixed_and_list_bf16<<<grid, block>>>(
        S.data_ptr<c10::BFloat16>(), P2.data_ptr<c10::BFloat16>(), C2.data_ptr<c10::BFloat16>(),
        D.data_ptr<float>(), (uint32_t*)idx_list.data_ptr<int32_t>(),
        (uint32_t*)list_count.data_ptr<int32_t>(), N, M, dim, kappa);

    // FALLBACK: FP32 direct formula
    const int blocks_list = ((int)total + threads - 1) / threads;
    direct_fp32_on_list<<<blocks_list, threads>>>(
        P.data_ptr<float>(), C.data_ptr<float>(), D.data_ptr<float>(),
        (uint32_t*)idx_list.data_ptr<int32_t>(),
        (uint32_t*)list_count.data_ptr<int32_t>(), dim, N, M);

    cudaDeviceSynchronize();
    return D;
}

std::tuple<torch::Tensor, int> pairwise_euclidean_bf16_fp32_with_stats(
    torch::Tensor P, torch::Tensor C, float kappa) {
    CHECK_INPUT(P);
    CHECK_INPUT(C);
    TORCH_CHECK(P.dtype() == torch::kFloat32, "P must be float32");
    TORCH_CHECK(C.dtype() == torch::kFloat32, "C must be float32");
    TORCH_CHECK(P.dim() == 2 && C.dim() == 2, "Inputs must be 2D");

    const int N = P.size(0);
    const int M = C.size(0);
    const int dim = P.size(1);
    TORCH_CHECK(C.size(1) == dim, "Dimension mismatch");

    const size_t total = (size_t)N * (size_t)M;
    TORCH_CHECK(total <= (size_t)UINT32_MAX, "N*M exceeds uint32 range");

    auto P_low = P.to(torch::kBFloat16);
    auto C_low = C.to(torch::kBFloat16);

    auto P2 = torch::empty({N}, torch::TensorOptions().dtype(torch::kBFloat16).device(P.device()));
    auto C2 = torch::empty({M}, torch::TensorOptions().dtype(torch::kBFloat16).device(C.device()));
    auto D  = torch::empty({N, M}, torch::TensorOptions().dtype(torch::kFloat32).device(P.device()));
    auto S  = torch::empty({N, M}, torch::TensorOptions().dtype(torch::kBFloat16).device(P.device()));

    const int threads = 256;
    compute_norms_bf16_uniform<<<(N + threads - 1) / threads, threads>>>(
        P_low.data_ptr<c10::BFloat16>(), P2.data_ptr<c10::BFloat16>(), N, dim);
    compute_norms_bf16_uniform<<<(M + threads - 1) / threads, threads>>>(
        C_low.data_ptr<c10::BFloat16>(), C2.data_ptr<c10::BFloat16>(), M, dim);

    cublasHandle_t handle = get_cublas_handle();
    sync_cublas_stream(handle);
    cublas_gemm_bf16(handle, P_low.data_ptr<c10::BFloat16>(), C_low.data_ptr<c10::BFloat16>(),
                     S.data_ptr<c10::BFloat16>(), N, M, dim);

    auto idx_list = torch::empty({(int64_t)total},
        torch::TensorOptions().dtype(torch::kInt32).device(P.device()));
    auto list_count = torch::zeros({1},
        torch::TensorOptions().dtype(torch::kInt32).device(P.device()));

    dim3 block(16, 16);
    dim3 grid((M + 15) / 16, (N + 15) / 16);
    form_mixed_and_list_bf16<<<grid, block>>>(
        S.data_ptr<c10::BFloat16>(), P2.data_ptr<c10::BFloat16>(), C2.data_ptr<c10::BFloat16>(),
        D.data_ptr<float>(), (uint32_t*)idx_list.data_ptr<int32_t>(),
        (uint32_t*)list_count.data_ptr<int32_t>(), N, M, dim, kappa);

    const int blocks_list = ((int)total + threads - 1) / threads;
    direct_fp32_on_list<<<blocks_list, threads>>>(
        P.data_ptr<float>(), C.data_ptr<float>(), D.data_ptr<float>(),
        (uint32_t*)idx_list.data_ptr<int32_t>(),
        (uint32_t*)list_count.data_ptr<int32_t>(), dim, N, M);

    cudaDeviceSynchronize();

    int32_t count;
    cudaMemcpy(&count, list_count.data_ptr<int32_t>(), sizeof(int32_t), cudaMemcpyDeviceToHost);
    return std::make_tuple(D, (int)count);
}

// ============================================================================
// FP32 -> FP32 DIRECT FALLBACK
// ============================================================================
torch::Tensor pairwise_euclidean_fp32_fp32(torch::Tensor P, torch::Tensor C, float kappa) {
    CHECK_INPUT(P);
    CHECK_INPUT(C);
    TORCH_CHECK(P.dtype() == torch::kFloat32, "P must be float32");
    TORCH_CHECK(C.dtype() == torch::kFloat32, "C must be float32");
    TORCH_CHECK(P.dim() == 2 && C.dim() == 2, "Inputs must be 2D");

    const int N = P.size(0);
    const int M = C.size(0);
    const int dim = P.size(1);
    TORCH_CHECK(C.size(1) == dim, "Dimension mismatch");

    const size_t total = (size_t)N * (size_t)M;
    TORCH_CHECK(total <= (size_t)UINT32_MAX, "N*M exceeds uint32 range");

    auto P2 = torch::empty({N}, P.options());
    auto C2 = torch::empty({M}, C.options());
    auto D  = torch::empty({N, M}, P.options());

    const int threads = 256;
    compute_norms_rowmajor_float<<<(N + threads - 1) / threads, threads>>>(
        P.data_ptr<float>(), P2.data_ptr<float>(), N, dim);
    compute_norms_rowmajor_float<<<(M + threads - 1) / threads, threads>>>(
        C.data_ptr<float>(), C2.data_ptr<float>(), M, dim);

    auto S = torch::mm(P, C.t());

    auto idx_list = torch::empty({(int64_t)total},
        torch::TensorOptions().dtype(torch::kInt32).device(P.device()));
    auto list_count = torch::zeros({1},
        torch::TensorOptions().dtype(torch::kInt32).device(P.device()));

    dim3 block(16, 16);
    dim3 grid((M + 15) / 16, (N + 15) / 16);
    form_mixed_and_list_fp32<<<grid, block>>>(
        S.data_ptr<float>(), P2.data_ptr<float>(), C2.data_ptr<float>(),
        D.data_ptr<float>(), (uint32_t*)idx_list.data_ptr<int32_t>(),
        (uint32_t*)list_count.data_ptr<int32_t>(), N, M, dim, kappa);

    // FALLBACK: FP32 direct formula
    const int blocks_list = ((int)total + threads - 1) / threads;
    direct_fp32_on_list<<<blocks_list, threads>>>(
        P.data_ptr<float>(), C.data_ptr<float>(), D.data_ptr<float>(),
        (uint32_t*)idx_list.data_ptr<int32_t>(),
        (uint32_t*)list_count.data_ptr<int32_t>(), dim, N, M);

    cudaDeviceSynchronize();
    return D;
}

std::tuple<torch::Tensor, int> pairwise_euclidean_fp32_fp32_with_stats(
    torch::Tensor P, torch::Tensor C, float kappa) {
    CHECK_INPUT(P);
    CHECK_INPUT(C);
    TORCH_CHECK(P.dtype() == torch::kFloat32, "P must be float32");
    TORCH_CHECK(C.dtype() == torch::kFloat32, "C must be float32");
    TORCH_CHECK(P.dim() == 2 && C.dim() == 2, "Inputs must be 2D");

    const int N = P.size(0);
    const int M = C.size(0);
    const int dim = P.size(1);
    TORCH_CHECK(C.size(1) == dim, "Dimension mismatch");

    const size_t total = (size_t)N * (size_t)M;
    TORCH_CHECK(total <= (size_t)UINT32_MAX, "N*M exceeds uint32 range");

    auto P2 = torch::empty({N}, P.options());
    auto C2 = torch::empty({M}, C.options());
    auto D  = torch::empty({N, M}, P.options());

    const int threads = 256;
    compute_norms_rowmajor_float<<<(N + threads - 1) / threads, threads>>>(
        P.data_ptr<float>(), P2.data_ptr<float>(), N, dim);
    compute_norms_rowmajor_float<<<(M + threads - 1) / threads, threads>>>(
        C.data_ptr<float>(), C2.data_ptr<float>(), M, dim);

    auto S = torch::mm(P, C.t());

    auto idx_list = torch::empty({(int64_t)total},
        torch::TensorOptions().dtype(torch::kInt32).device(P.device()));
    auto list_count = torch::zeros({1},
        torch::TensorOptions().dtype(torch::kInt32).device(P.device()));

    dim3 block(16, 16);
    dim3 grid((M + 15) / 16, (N + 15) / 16);
    form_mixed_and_list_fp32<<<grid, block>>>(
        S.data_ptr<float>(), P2.data_ptr<float>(), C2.data_ptr<float>(),
        D.data_ptr<float>(), (uint32_t*)idx_list.data_ptr<int32_t>(),
        (uint32_t*)list_count.data_ptr<int32_t>(), N, M, dim, kappa);

    const int blocks_list = ((int)total + threads - 1) / threads;
    direct_fp32_on_list<<<blocks_list, threads>>>(
        P.data_ptr<float>(), C.data_ptr<float>(), D.data_ptr<float>(),
        (uint32_t*)idx_list.data_ptr<int32_t>(),
        (uint32_t*)list_count.data_ptr<int32_t>(), dim, N, M);

    cudaDeviceSynchronize();

    int32_t count;
    cudaMemcpy(&count, list_count.data_ptr<int32_t>(), sizeof(int32_t), cudaMemcpyDeviceToHost);
    return std::make_tuple(D, (int)count);
}