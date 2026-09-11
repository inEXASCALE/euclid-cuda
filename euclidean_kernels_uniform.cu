/*
 * Uniform Precision Kernels Implementation
 * Pure low-precision, no fallback
 */
#include "euclidean_kernels_uniform.cuh"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

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
// FP16 UNIFORM KERNELS
// ============================================================================

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

__global__ void form_distance_fp16_uniform(
    const c10::Half* S, const c10::Half* P2, const c10::Half* C2,
    c10::Half* D, int N, int M) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || j >= M) return;
    
    __half s_h = to_cuda_half(S[i * M + j]);
    __half p2_h = to_cuda_half(P2[i]);
    __half c2_h = to_cuda_half(C2[j]);
    
    float s = __half2float(s_h);
    float p2 = __half2float(p2_h);
    float c2 = __half2float(c2_h);
    
    float d = p2 - 2.f * s + c2;
    d = fmaxf(d, 0.f);
    
    D[i * M + j] = __float2half_rn(d);
}

// ============================================================================
// BF16 UNIFORM KERNELS
// ============================================================================

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

__global__ void form_distance_bf16_uniform(
    const c10::BFloat16* S, const c10::BFloat16* P2, const c10::BFloat16* C2,
    c10::BFloat16* D, int N, int M) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || j >= M) return;
    
    __nv_bfloat16 s_bf = to_cuda_bfloat16(S[i * M + j]);
    __nv_bfloat16 p2_bf = to_cuda_bfloat16(P2[i]);
    __nv_bfloat16 c2_bf = to_cuda_bfloat16(C2[j]);
    
    float s = __bfloat162float(s_bf);
    float p2 = __bfloat162float(p2_bf);
    float c2 = __bfloat162float(c2_bf);
    
    float d = p2 - 2.f * s + c2;
    d = fmaxf(d, 0.f);
    
    D[i * M + j] = __float2bfloat16_rn(d);
}

// ============================================================================
// TF32 UNIFORM KERNELS
// ============================================================================

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

__global__ void form_distance_tf32_uniform(
    const float* S, const float* P2, const float* C2,
    float* D, int N, int M) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || j >= M) return;
    
    float s = S[i * M + j];
    float d = P2[i] - 2.f * s + C2[j];
    D[i * M + j] = fmaxf(d, 0.f);
}

// ============================================================================
// HOST IMPLEMENTATIONS
// ============================================================================
torch::Tensor pairwise_euclidean_uniform_fp16(torch::Tensor P, torch::Tensor C) {
    CHECK_INPUT(P);
    CHECK_INPUT(C);
    TORCH_CHECK(P.dtype() == torch::kFloat16, "P must be float16");
    TORCH_CHECK(C.dtype() == torch::kFloat16, "C must be float16");
    TORCH_CHECK(P.dim() == 2 && C.dim() == 2, "Inputs must be 2D");
    
    const int N = P.size(0);
    const int M = C.size(0);
    const int dim = P.size(1);
    TORCH_CHECK(C.size(1) == dim, "Dimension mismatch");
    
    // Allocate outputs in FP16
    auto P2 = torch::empty({N}, P.options());
    auto C2 = torch::empty({M}, C.options());
    auto D = torch::empty({N, M}, P.options());
    
    const int threads = 256;
    
    compute_norms_fp16_uniform<<<(N + threads - 1) / threads, threads>>>(
        P.data_ptr<c10::Half>(), P2.data_ptr<c10::Half>(), N, dim);
    compute_norms_fp16_uniform<<<(M + threads - 1) / threads, threads>>>(
        C.data_ptr<c10::Half>(), C2.data_ptr<c10::Half>(), M, dim);
    
    auto S = torch::mm(P, C.t());   
    
    dim3 block(16, 16);
    dim3 grid((M + 15) / 16, (N + 15) / 16);
    form_distance_fp16_uniform<<<grid, block>>>(
        S.data_ptr<c10::Half>(),
        P2.data_ptr<c10::Half>(),
        C2.data_ptr<c10::Half>(),
        D.data_ptr<c10::Half>(),
        N, M);
    
    cudaDeviceSynchronize();
    return D;
}

torch::Tensor pairwise_euclidean_uniform_bf16(torch::Tensor P, torch::Tensor C) {
    CHECK_INPUT(P);
    CHECK_INPUT(C);
    TORCH_CHECK(P.dtype() == torch::kBFloat16, "P must be bfloat16");
    TORCH_CHECK(C.dtype() == torch::kBFloat16, "C must be bfloat16");
    TORCH_CHECK(P.dim() == 2 && C.dim() == 2, "Inputs must be 2D");
    
    const int N = P.size(0);
    const int M = C.size(0);
    const int dim = P.size(1);
    TORCH_CHECK(C.size(1) == dim, "Dimension mismatch");
    
    auto P2 = torch::empty({N}, P.options());
    auto C2 = torch::empty({M}, C.options());
    auto D = torch::empty({N, M}, P.options());
    
    const int threads = 256;
    
    compute_norms_bf16_uniform<<<(N + threads - 1) / threads, threads>>>(
        P.data_ptr<c10::BFloat16>(), P2.data_ptr<c10::BFloat16>(), N, dim);
    compute_norms_bf16_uniform<<<(M + threads - 1) / threads, threads>>>(
        C.data_ptr<c10::BFloat16>(), C2.data_ptr<c10::BFloat16>(), M, dim);
    
    auto S = torch::mm(P, C.t());  
    
    dim3 block(16, 16);
    dim3 grid((M + 15) / 16, (N + 15) / 16);
    form_distance_bf16_uniform<<<grid, block>>>(
        S.data_ptr<c10::BFloat16>(),
        P2.data_ptr<c10::BFloat16>(),
        C2.data_ptr<c10::BFloat16>(),
        D.data_ptr<c10::BFloat16>(),
        N, M);
    
    cudaDeviceSynchronize();
    return D;
}

torch::Tensor pairwise_euclidean_uniform_tf32(torch::Tensor P, torch::Tensor C) {
    CHECK_INPUT(P);
    CHECK_INPUT(C);
    TORCH_CHECK(P.dtype() == torch::kFloat32, "P must be float32");
    TORCH_CHECK(C.dtype() == torch::kFloat32, "C must be float32");
    TORCH_CHECK(P.dim() == 2 && C.dim() == 2, "Inputs must be 2D");
    
    const int N = P.size(0);
    const int M = C.size(0);
    const int dim = P.size(1);
    TORCH_CHECK(C.size(1) == dim, "Dimension mismatch");
    
    auto P2 = torch::empty({N}, P.options());
    auto C2 = torch::empty({M}, C.options());
    auto D = torch::empty({N, M}, P.options());
    
    const int threads = 256;
    
    compute_norms_tf32_uniform<<<(N + threads - 1) / threads, threads>>>(
        P.data_ptr<float>(), P2.data_ptr<float>(), N, dim);
    compute_norms_tf32_uniform<<<(M + threads - 1) / threads, threads>>>(
        C.data_ptr<float>(), C2.data_ptr<float>(), M, dim);
    
    // GEMM with TF32 (PyTorch automatically uses TF32 on A100+)
    auto S = torch::mm(P, C.t());
    
    dim3 block(16, 16);
    dim3 grid((M + 15) / 16, (N + 15) / 16);
    form_distance_tf32_uniform<<<grid, block>>>(
        S.data_ptr<float>(),
        P2.data_ptr<float>(),
        C2.data_ptr<float>(),
        D.data_ptr<float>(),
        N, M);
    
    cudaDeviceSynchronize();
    return D;
}




// ============================================================================
// DIRECT FORMULA KERNELS - FP16
// More stable for near points, no expand/GEMM needed
// ============================================================================

__global__ void pairwise_distance_direct_fp16(
    const c10::Half* P, const c10::Half* C, c10::Half* D,
    int N, int M, int dim) {
    
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= N || j >= M) return;
    
    const c10::Half* p = P + i * dim;
    const c10::Half* c = C + j * dim;
    
    // Direct formula: sum((p_k - c_k)^2) in FP16
    float acc = 0.0f;  // Use FP32 accumulator for better precision
    for (int k = 0; k < dim; ++k) {
        __half ph = to_cuda_half(p[k]);
        __half ch = to_cuda_half(c[k]);
        float pf = __half2float(ph);
        float cf = __half2float(ch);
        float diff = pf - cf;
        acc += diff * diff;
    }
    
    D[i * M + j] = __float2half_rn(acc);
}

torch::Tensor pairwise_euclidean_uniform_fp16_direct(torch::Tensor P, torch::Tensor C) {
    CHECK_INPUT(P);
    CHECK_INPUT(C);
    TORCH_CHECK(P.dtype() == torch::kFloat16, "P must be float16");
    TORCH_CHECK(C.dtype() == torch::kFloat16, "C must be float16");
    TORCH_CHECK(P.dim() == 2 && C.dim() == 2, "Inputs must be 2D");
    
    const int N = P.size(0);
    const int M = C.size(0);
    const int dim = P.size(1);
    TORCH_CHECK(C.size(1) == dim, "Dimension mismatch");
    
    auto D = torch::empty({N, M}, P.options());
    
    dim3 block(16, 16);
    dim3 grid((M + 15) / 16, (N + 15) / 16);
    
    pairwise_distance_direct_fp16<<<grid, block>>>(
        P.data_ptr<c10::Half>(),
        C.data_ptr<c10::Half>(),
        D.data_ptr<c10::Half>(),
        N, M, dim);
    
    cudaDeviceSynchronize();
    return D;
}

// ============================================================================
// DIRECT FORMULA KERNELS - BF16
// ============================================================================

__global__ void pairwise_distance_direct_bf16(
    const c10::BFloat16* P, const c10::BFloat16* C, c10::BFloat16* D,
    int N, int M, int dim) {
    
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= N || j >= M) return;
    
    const c10::BFloat16* p = P + i * dim;
    const c10::BFloat16* c = C + j * dim;
    
    // Direct formula: sum((p_k - c_k)^2) in BF16
    float acc = 0.0f;
    for (int k = 0; k < dim; ++k) {
        __nv_bfloat16 pb = to_cuda_bfloat16(p[k]);
        __nv_bfloat16 cb = to_cuda_bfloat16(c[k]);
        float pf = __bfloat162float(pb);
        float cf = __bfloat162float(cb);
        float diff = pf - cf;
        acc += diff * diff;
    }
    
    D[i * M + j] = __float2bfloat16_rn(acc);
}

torch::Tensor pairwise_euclidean_uniform_bf16_direct(torch::Tensor P, torch::Tensor C) {
    CHECK_INPUT(P);
    CHECK_INPUT(C);
    TORCH_CHECK(P.dtype() == torch::kBFloat16, "P must be bfloat16");
    TORCH_CHECK(C.dtype() == torch::kBFloat16, "C must be bfloat16");
    TORCH_CHECK(P.dim() == 2 && C.dim() == 2, "Inputs must be 2D");
    
    const int N = P.size(0);
    const int M = C.size(0);
    const int dim = P.size(1);
    TORCH_CHECK(C.size(1) == dim, "Dimension mismatch");
    
    auto D = torch::empty({N, M}, P.options());
    
    dim3 block(16, 16);
    dim3 grid((M + 15) / 16, (N + 15) / 16);
    
    pairwise_distance_direct_bf16<<<grid, block>>>(
        P.data_ptr<c10::BFloat16>(),
        C.data_ptr<c10::BFloat16>(),
        D.data_ptr<c10::BFloat16>(),
        N, M, dim);
    
    cudaDeviceSynchronize();
    return D;
}

// ============================================================================
// DIRECT FORMULA KERNELS - TF32 (FP32 storage)
// ============================================================================

__global__ void pairwise_distance_direct_tf32(
    const float* P, const float* C, float* D,
    int N, int M, int dim) {
    
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= N || j >= M) return;
    
    const float* p = P + i * dim;
    const float* c = C + j * dim;
    
    // Direct formula: sum((p_k - c_k)^2) in FP32
    float acc = 0.0f;
    for (int k = 0; k < dim; ++k) {
        float diff = p[k] - c[k];
        acc += diff * diff;
    }
    
    D[i * M + j] = acc;
}

torch::Tensor pairwise_euclidean_uniform_tf32_direct(torch::Tensor P, torch::Tensor C) {
    CHECK_INPUT(P);
    CHECK_INPUT(C);
    TORCH_CHECK(P.dtype() == torch::kFloat32, "P must be float32");
    TORCH_CHECK(C.dtype() == torch::kFloat32, "C must be float32");
    TORCH_CHECK(P.dim() == 2 && C.dim() == 2, "Inputs must be 2D");
    
    const int N = P.size(0);
    const int M = C.size(0);
    const int dim = P.size(1);
    TORCH_CHECK(C.size(1) == dim, "Dimension mismatch");
    
    auto D = torch::empty({N, M}, P.options());
    
    dim3 block(16, 16);
    dim3 grid((M + 15) / 16, (N + 15) / 16);
    
    pairwise_distance_direct_tf32<<<grid, block>>>(
        P.data_ptr<float>(),
        C.data_ptr<float>(),
        D.data_ptr<float>(),
        N, M, dim);
    
    cudaDeviceSynchronize();
    return D;
}



// ============================================================================
// DIRECT FORMULA KERNELS - FP32
// ============================================================================

__global__ void pairwise_distance_direct_fp32(
    const float* P, const float* C, float* D,
    int N, int M, int dim) {
    
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= N || j >= M) return;
    
    const float* p = P + i * dim;
    const float* c = C + j * dim;
    
    // Direct formula: sum((p_k - c_k)^2) in FP32
    float acc = 0.0f;
    for (int k = 0; k < dim; ++k) {
        float diff = p[k] - c[k];
        acc = fmaf(diff, diff, acc);  // FMA for better precision
    }
    
    D[i * M + j] = acc;
}

torch::Tensor pairwise_euclidean_uniform_fp32_direct(torch::Tensor P, torch::Tensor C) {
    CHECK_INPUT(P);
    CHECK_INPUT(C);
    TORCH_CHECK(P.dtype() == torch::kFloat32, "P must be float32");
    TORCH_CHECK(C.dtype() == torch::kFloat32, "C must be float32");
    TORCH_CHECK(P.dim() == 2 && C.dim() == 2, "Inputs must be 2D");
    
    const int N = P.size(0);
    const int M = C.size(0);
    const int dim = P.size(1);
    TORCH_CHECK(C.size(1) == dim, "Dimension mismatch");
    
    auto D = torch::empty({N, M}, P.options());
    
    dim3 block(16, 16);
    dim3 grid((M + 15) / 16, (N + 15) / 16);
    
    pairwise_distance_direct_fp32<<<grid, block>>>(
        P.data_ptr<float>(),
        C.data_ptr<float>(),
        D.data_ptr<float>(),
        N, M, dim);
    
    cudaDeviceSynchronize();
    return D;
}

// ============================================================================
// DIRECT FORMULA KERNELS - FP64
// Highest precision, slowest
// ============================================================================

__global__ void pairwise_distance_direct_fp64(
    const double* P, const double* C, double* D,
    int N, int M, int dim) {
    
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= N || j >= M) return;
    
    const double* p = P + i * dim;
    const double* c = C + j * dim;
    
    // Direct formula: sum((p_k - c_k)^2) in FP64
    double acc = 0.0;
    for (int k = 0; k < dim; ++k) {
        double diff = p[k] - c[k];
        acc = fma(diff, diff, acc);  // FMA in FP64
    }
    
    D[i * M + j] = acc;
}

torch::Tensor pairwise_euclidean_uniform_fp64_direct(torch::Tensor P, torch::Tensor C) {
    CHECK_INPUT(P);
    CHECK_INPUT(C);
    TORCH_CHECK(P.dtype() == torch::kFloat64, "P must be float64");
    TORCH_CHECK(C.dtype() == torch::kFloat64, "C must be float64");
    TORCH_CHECK(P.dim() == 2 && C.dim() == 2, "Inputs must be 2D");
    
    const int N = P.size(0);
    const int M = C.size(0);
    const int dim = P.size(1);
    TORCH_CHECK(C.size(1) == dim, "Dimension mismatch");
    
    auto D = torch::empty({N, M}, P.options());
    
    dim3 block(16, 16);
    dim3 grid((M + 15) / 16, (N + 15) / 16);
    
    pairwise_distance_direct_fp64<<<grid, block>>>(
        P.data_ptr<double>(),
        C.data_ptr<double>(),
        D.data_ptr<double>(),
        N, M, dim);
    
    cudaDeviceSynchronize();
    return D;
}