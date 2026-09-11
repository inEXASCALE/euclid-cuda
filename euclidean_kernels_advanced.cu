/*
 * PyTorch CUDA Extension: Advanced Mixed-Precision Euclidean Distance
 * FIXED: Proper type conversions for FP16/BF16
 */
#include "euclidean_kernels_advanced.cuh"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// ============================================================================
// TYPE CONVERSION HELPERS
// ============================================================================

// PyTorch's at::Half <-> CUDA's __half
__device__ __forceinline__ __half to_cuda_half(const c10::Half& x) {
    return *reinterpret_cast<const __half*>(&x);
}

__device__ __forceinline__ c10::Half from_cuda_half(const __half& x) {
    return *reinterpret_cast<const c10::Half*>(&x);
}

// PyTorch's at::BFloat16 <-> CUDA's __nv_bfloat16
__device__ __forceinline__ __nv_bfloat16 to_cuda_bfloat16(const c10::BFloat16& x) {
    return *reinterpret_cast<const __nv_bfloat16*>(&x);
}

__device__ __forceinline__ c10::BFloat16 from_cuda_bfloat16(const __nv_bfloat16& x) {
    return *reinterpret_cast<const c10::BFloat16*>(&x);
}

// ============================================================================
// NORM KERNELS
// ============================================================================
__global__ void compute_norms_fp16(const c10::Half* X, float* X2, int N, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    const c10::Half* row = X + idx * dim;
    float s = 0.f;
    for (int k = 0; k < dim; ++k) {
        __half h = to_cuda_half(row[k]);
        float v = __half2float(h);
        s = fmaf(v, v, s);
    }
    X2[idx] = s;
}

__global__ void compute_norms_bf16(const c10::BFloat16* X, float* X2, int N, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    const c10::BFloat16* row = X + idx * dim;
    float s = 0.f;
    for (int k = 0; k < dim; ++k) {
        __nv_bfloat16 bf = to_cuda_bfloat16(row[k]);
        float v = __bfloat162float(bf);
        s = fmaf(v, v, s);
    }
    X2[idx] = s;
}

__global__ void compute_norms_rowmajor_float(const float* X, float* X2, int N, int dim) {
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

__global__ void compute_norms_rowmajor_double(const double* X, double* X2, int N, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    const double* row = X + idx * dim;
    double s = 0.0;
    for (int k = 0; k < dim; ++k) {
        double v = row[k];
        s += v * v;
    }
    X2[idx] = s;
}

// ============================================================================
// DISTANCE FORMATION KERNELS
// ============================================================================
__global__ void form_expanded_rowmajor_double_to_float(
    const double* S, const float* P2, const float* C2,
    float* D, int N, int M) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || j >= M) return;
    
    double s = S[i * M + j];
    double d = (double)P2[i] - 2.0 * s + (double)C2[j];
    D[i * M + j] = (float)fmax(d, 0.0);
}

__global__ void form_mixed_and_list_double(
    const double* S, const float* P2, const float* C2,
    float* D, uint32_t* idx_list, uint32_t* list_count,
    int N, int M, float gamma_n, float kappa) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || j >= M) return;
    
    size_t pos = i * M + j;
    double s = S[pos];
    double d_fp64 = (double)P2[i] - 2.0 * s + (double)C2[j];
    float d = (float)fmax(d_fp64, 0.0);
    D[pos] = d;
    
    float err_floor = kappa * 2.f * gamma_n * (P2[i] + C2[j]);
    if (d <= err_floor) {
        uint32_t p = atomicAdd(list_count, 1u);
        idx_list[p] = (uint32_t)pos;
    }
}

__global__ void form_mixed_and_list_double_output(
    const double* S, const double* P2, const double* C2,
    double* D, uint32_t* idx_list, uint32_t* list_count,
    int N, int M, double gamma_n, double kappa) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || j >= M) return;
    
    size_t pos = i * M + j;
    double s = S[pos];
    double d = P2[i] - 2.0 * s + C2[j];
    d = fmax(d, 0.0);
    D[pos] = d;
    
    double err_floor = kappa * 2.0 * gamma_n * (P2[i] + C2[j]);
    if (d <= err_floor) {
        uint32_t p = atomicAdd(list_count, 1u);
        idx_list[p] = (uint32_t)pos;
    }
}

// ============================================================================
// FALLBACK KERNELS - FIXED TYPE CONVERSIONS
// ============================================================================
__global__ void direct_on_list_fp16(
    const c10::Half* P, const c10::Half* C, float* D,
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
    double acc = 0.0;
    for (int k = 0; k < dim; ++k) {
        __half p_h = to_cuda_half(p[k]);
        __half c_h = to_cuda_half(c[k]);
        double p_val = (double)__half2float(p_h);
        double c_val = (double)__half2float(c_h);
        double diff = p_val - c_val;
        acc += diff * diff;
    }
    D[i * M + j] = (float)acc;
}

__global__ void direct_on_list_bf16(
    const c10::BFloat16* P, const c10::BFloat16* C, float* D,
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
    double acc = 0.0;
    for (int k = 0; k < dim; ++k) {
        __nv_bfloat16 p_bf = to_cuda_bfloat16(p[k]);
        __nv_bfloat16 c_bf = to_cuda_bfloat16(c[k]);
        double p_val = (double)__bfloat162float(p_bf);
        double c_val = (double)__bfloat162float(c_bf);
        double diff = p_val - c_val;
        acc += diff * diff;
    }
    D[i * M + j] = (float)acc;
}

__global__ void direct_on_list_fp64(
    const double* P, const double* C, double* D,
    const uint32_t* idx_list, const uint32_t* list_count,
    int dim, int N, int M) {
    uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t cnt = *list_count;
    if (t >= cnt) return;
    uint32_t idx = idx_list[t];
    int i = idx / M;
    int j = idx % M;
    if (i >= N || j >= M) return;
    
    const double* p = P + i * dim;
    const double* c = C + j * dim;
    double acc = 0.0;
    for (int k = 0; k < dim; ++k) {
        double diff = p[k] - c[k];
        acc += diff * diff;
    }
    D[i * M + j] = acc;
}

// ============================================================================
// HOST IMPLEMENTATIONS
// ============================================================================
torch::Tensor pairwise_euclidean_fp64_fp16(torch::Tensor P, torch::Tensor C, float kappa) {
    CHECK_INPUT(P);
    CHECK_INPUT(C);
    TORCH_CHECK(P.dtype() == torch::kFloat16, "P must be float16");
    TORCH_CHECK(C.dtype() == torch::kFloat16, "C must be float16");
    TORCH_CHECK(P.dim() == 2 && C.dim() == 2, "Inputs must be 2D");
    
    const int N = P.size(0);
    const int M = C.size(0);
    const int dim = P.size(1);
    TORCH_CHECK(C.size(1) == dim, "Dimension mismatch");

    const float u = 1.1920929e-7f;
    const float gamma_n = dim * u;

    auto P_f32 = P.to(torch::kFloat32);
    auto C_f32 = C.to(torch::kFloat32);
    
    auto P2 = torch::empty({N}, torch::TensorOptions().dtype(torch::kFloat32).device(P.device()));
    auto C2 = torch::empty({M}, torch::TensorOptions().dtype(torch::kFloat32).device(C.device()));
    auto D = torch::empty({N, M}, torch::TensorOptions().dtype(torch::kFloat32).device(P.device()));

    const int threads = 256;
    compute_norms_rowmajor_float<<<(N + threads - 1) / threads, threads>>>(
        P_f32.data_ptr<float>(), P2.data_ptr<float>(), N, dim);
    compute_norms_rowmajor_float<<<(M + threads - 1) / threads, threads>>>(
        C_f32.data_ptr<float>(), C2.data_ptr<float>(), M, dim);

    auto S = torch::mm(P_f32.to(torch::kFloat64), C_f32.to(torch::kFloat64).t());

    const size_t total = (size_t)N * (size_t)M;
    auto idx_list = torch::empty({(int64_t)total}, 
        torch::TensorOptions().dtype(torch::kInt32).device(P.device()));
    auto list_count = torch::zeros({1}, 
        torch::TensorOptions().dtype(torch::kInt32).device(P.device()));

    dim3 block(16, 16);
    dim3 grid((M + 15) / 16, (N + 15) / 16);
    form_mixed_and_list_double<<<grid, block>>>(
        S.data_ptr<double>(), P2.data_ptr<float>(), C2.data_ptr<float>(),
        D.data_ptr<float>(),
        (uint32_t*)idx_list.data_ptr<int32_t>(),
        (uint32_t*)list_count.data_ptr<int32_t>(),
        N, M, gamma_n, kappa);

    const int blocks_list = ((int)total + threads - 1) / threads;
    direct_on_list_fp16<<<blocks_list, threads>>>(
        P.data_ptr<c10::Half>(),              // FIXED: use c10::Half directly
        C.data_ptr<c10::Half>(),
        D.data_ptr<float>(),
        (uint32_t*)idx_list.data_ptr<int32_t>(),
        (uint32_t*)list_count.data_ptr<int32_t>(),
        dim, N, M);

    cudaDeviceSynchronize();
    return D;
}

torch::Tensor pairwise_euclidean_fp64_bf16(torch::Tensor P, torch::Tensor C, float kappa) {
    CHECK_INPUT(P);
    CHECK_INPUT(C);
    TORCH_CHECK(P.dtype() == torch::kBFloat16, "P must be bfloat16");
    TORCH_CHECK(C.dtype() == torch::kBFloat16, "C must be bfloat16");
    TORCH_CHECK(P.dim() == 2 && C.dim() == 2, "Inputs must be 2D");
    
    const int N = P.size(0);
    const int M = C.size(0);
    const int dim = P.size(1);
    TORCH_CHECK(C.size(1) == dim, "Dimension mismatch");

    const float u = 1.1920929e-7f;
    const float gamma_n = dim * u;

    auto P_f32 = P.to(torch::kFloat32);
    auto C_f32 = C.to(torch::kFloat32);
    
    auto P2 = torch::empty({N}, torch::TensorOptions().dtype(torch::kFloat32).device(P.device()));
    auto C2 = torch::empty({M}, torch::TensorOptions().dtype(torch::kFloat32).device(C.device()));
    auto D = torch::empty({N, M}, torch::TensorOptions().dtype(torch::kFloat32).device(P.device()));

    const int threads = 256;
    compute_norms_rowmajor_float<<<(N + threads - 1) / threads, threads>>>(
        P_f32.data_ptr<float>(), P2.data_ptr<float>(), N, dim);
    compute_norms_rowmajor_float<<<(M + threads - 1) / threads, threads>>>(
        C_f32.data_ptr<float>(), C2.data_ptr<float>(), M, dim);

    auto S = torch::mm(P_f32.to(torch::kFloat64), C_f32.to(torch::kFloat64).t());

    const size_t total = (size_t)N * (size_t)M;
    auto idx_list = torch::empty({(int64_t)total}, 
        torch::TensorOptions().dtype(torch::kInt32).device(P.device()));
    auto list_count = torch::zeros({1}, 
        torch::TensorOptions().dtype(torch::kInt32).device(P.device()));

    dim3 block(16, 16);
    dim3 grid((M + 15) / 16, (N + 15) / 16);
    form_mixed_and_list_double<<<grid, block>>>(
        S.data_ptr<double>(), P2.data_ptr<float>(), C2.data_ptr<float>(),
        D.data_ptr<float>(),
        (uint32_t*)idx_list.data_ptr<int32_t>(),
        (uint32_t*)list_count.data_ptr<int32_t>(),
        N, M, gamma_n, kappa);

    const int blocks_list = ((int)total + threads - 1) / threads;
    direct_on_list_bf16<<<blocks_list, threads>>>(
        P.data_ptr<c10::BFloat16>(),          // FIXED: use c10::BFloat16 directly
        C.data_ptr<c10::BFloat16>(),
        D.data_ptr<float>(),
        (uint32_t*)idx_list.data_ptr<int32_t>(),
        (uint32_t*)list_count.data_ptr<int32_t>(),
        dim, N, M);

    cudaDeviceSynchronize();
    return D;
}

torch::Tensor pairwise_euclidean_fp64_tf32(torch::Tensor P, torch::Tensor C, float kappa) {
    CHECK_INPUT(P);
    CHECK_INPUT(C);
    TORCH_CHECK(P.dtype() == torch::kFloat32, "P must be float32");
    TORCH_CHECK(C.dtype() == torch::kFloat32, "C must be float32");
    TORCH_CHECK(P.dim() == 2 && C.dim() == 2, "Inputs must be 2D");
    
    const int N = P.size(0);
    const int M = C.size(0);
    const int dim = P.size(1);
    TORCH_CHECK(C.size(1) == dim, "Dimension mismatch");

    const double u = 2.220446049250313e-16;
    const double gamma_n = dim * u;

    auto P_f64 = P.to(torch::kFloat64);
    auto C_f64 = C.to(torch::kFloat64);
    
    auto P2 = torch::empty({N}, torch::TensorOptions().dtype(torch::kFloat64).device(P.device()));
    auto C2 = torch::empty({M}, torch::TensorOptions().dtype(torch::kFloat64).device(C.device()));
    auto D = torch::empty({N, M}, torch::TensorOptions().dtype(torch::kFloat64).device(P.device()));

    const int threads = 256;
    compute_norms_rowmajor_double<<<(N + threads - 1) / threads, threads>>>(
        P_f64.data_ptr<double>(), P2.data_ptr<double>(), N, dim);
    compute_norms_rowmajor_double<<<(M + threads - 1) / threads, threads>>>(
        C_f64.data_ptr<double>(), C2.data_ptr<double>(), M, dim);

    // TF32 GEMM (fast but less accurate)
    auto S_tf32 = torch::mm(P, C.t()).to(torch::kFloat64);

    const size_t total = (size_t)N * (size_t)M;
    auto idx_list = torch::empty({(int64_t)total}, 
        torch::TensorOptions().dtype(torch::kInt32).device(P.device()));
    auto list_count = torch::zeros({1}, 
        torch::TensorOptions().dtype(torch::kInt32).device(P.device()));

    dim3 block(16, 16);
    dim3 grid((M + 15) / 16, (N + 15) / 16);
    form_mixed_and_list_double_output<<<grid, block>>>(
        S_tf32.data_ptr<double>(), P2.data_ptr<double>(), C2.data_ptr<double>(),
        D.data_ptr<double>(),
        (uint32_t*)idx_list.data_ptr<int32_t>(),
        (uint32_t*)list_count.data_ptr<int32_t>(),
        N, M, gamma_n, kappa);

    const int blocks_list = ((int)total + threads - 1) / threads;
    direct_on_list_fp64<<<blocks_list, threads>>>(
        P_f64.data_ptr<double>(), C_f64.data_ptr<double>(), D.data_ptr<double>(),
        (uint32_t*)idx_list.data_ptr<int32_t>(),
        (uint32_t*)list_count.data_ptr<int32_t>(),
        dim, N, M);

    cudaDeviceSynchronize();
    return D;
}

torch::Tensor pairwise_euclidean_fp64_fp32_gemm(torch::Tensor P, torch::Tensor C, float kappa) {
    CHECK_INPUT(P);
    CHECK_INPUT(C);
    TORCH_CHECK(P.dtype() == torch::kFloat64, "P must be float64");
    TORCH_CHECK(C.dtype() == torch::kFloat64, "C must be float64");
    TORCH_CHECK(P.dim() == 2 && C.dim() == 2, "Inputs must be 2D");
    
    const int N = P.size(0);
    const int M = C.size(0);
    const int dim = P.size(1);
    TORCH_CHECK(C.size(1) == dim, "Dimension mismatch");

    const double u = 2.220446049250313e-16;
    const double gamma_n = dim * u;

    auto P2 = torch::empty({N}, P.options());
    auto C2 = torch::empty({M}, C.options());
    auto D = torch::empty({N, M}, P.options());

    const int threads = 256;
    compute_norms_rowmajor_double<<<(N + threads - 1) / threads, threads>>>(
        P.data_ptr<double>(), P2.data_ptr<double>(), N, dim);
    compute_norms_rowmajor_double<<<(M + threads - 1) / threads, threads>>>(
        C.data_ptr<double>(), C2.data_ptr<double>(), M, dim);

    // FP32 GEMM (faster)
    auto P_f32 = P.to(torch::kFloat32);
    auto C_f32 = C.to(torch::kFloat32);
    auto S_f32 = torch::mm(P_f32, C_f32.t());
    auto S = S_f32.to(torch::kFloat64);

    const size_t total = (size_t)N * (size_t)M;
    auto idx_list = torch::empty({(int64_t)total}, 
        torch::TensorOptions().dtype(torch::kInt32).device(P.device()));
    auto list_count = torch::zeros({1}, 
        torch::TensorOptions().dtype(torch::kInt32).device(P.device()));

    dim3 block(16, 16);
    dim3 grid((M + 15) / 16, (N + 15) / 16);
    form_mixed_and_list_double_output<<<grid, block>>>(
        S.data_ptr<double>(), P2.data_ptr<double>(), C2.data_ptr<double>(),
        D.data_ptr<double>(),
        (uint32_t*)idx_list.data_ptr<int32_t>(),
        (uint32_t*)list_count.data_ptr<int32_t>(),
        N, M, gamma_n, kappa);

    const int blocks_list = ((int)total + threads - 1) / threads;
    direct_on_list_fp64<<<blocks_list, threads>>>(
        P.data_ptr<double>(), C.data_ptr<double>(), D.data_ptr<double>(),
        (uint32_t*)idx_list.data_ptr<int32_t>(),
        (uint32_t*)list_count.data_ptr<int32_t>(),
        dim, N, M);

    cudaDeviceSynchronize();
    return D;
}

int get_last_fallback_count(torch::Tensor list_count_tensor) {
    uint32_t count;
    cudaMemcpy(&count, list_count_tensor.data_ptr<int32_t>(), 
               sizeof(uint32_t), cudaMemcpyDeviceToHost);
    return (int)count;
}