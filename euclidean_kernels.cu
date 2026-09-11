/*
 * PyTorch CUDA Extension: Fast Euclidean Distance
 * Using PyTorch's GEMM (simpler and correct)
 */
#include "euclidean_kernels.cuh"
#include <ATen/cuda/CUDABlas.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// ============================================================================
// KERNELS (same as before)
// ============================================================================

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

__global__ void form_expanded_rowmajor_float(const float* S, const float* P2, const float* C2,
                                             float* D, int N, int M) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || j >= M) return;
    float s = S[i * M + j];
    float d = P2[i] - 2.f * s + C2[j];
    D[i * M + j] = fmaxf(d, 0.f);
}

__global__ void form_expanded_rowmajor_double(const double* S, const double* P2, const double* C2,
                                              double* D, int N, int M) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || j >= M) return;
    double s = S[i * M + j];
    double d = P2[i] - 2.0 * s + C2[j];
    D[i * M + j] = (d < 0.0 ? 0.0 : d);
}

__global__ void form_mixed_and_list_rowmajor(const float* S, const float* P2, const float* C2,
                                             float* D, uint32_t* idx_list, uint32_t* list_count,
                                             int N, int M, float gamma_n, float kappa) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || j >= M) return;
    size_t pos = i * M + j;
    float s = S[pos];
    float d = P2[i] - 2.f * s + C2[j];
    d = fmaxf(d, 0.f);
    D[pos] = d;
    float err_floor = kappa * 2.f * gamma_n * (P2[i] + C2[j]);
    if (d <= err_floor) {
        uint32_t p = atomicAdd(list_count, 1u);
        idx_list[p] = (uint32_t)pos;
    }
}


// ============================================================================
// HOST FUNCTIONS - Using PyTorch's mm()
// ============================================================================

torch::Tensor pairwise_euclidean_single(torch::Tensor P, torch::Tensor C) {
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

    // Step 1: Compute norms with custom kernel
    const int threads = 256;
    compute_norms_rowmajor_float<<<(N + threads - 1) / threads, threads>>>(
        P.data_ptr<float>(), P2.data_ptr<float>(), N, dim);
    compute_norms_rowmajor_float<<<(M + threads - 1) / threads, threads>>>(
        C.data_ptr<float>(), C2.data_ptr<float>(), M, dim);

    // Step 2: Use PyTorch's GEMM (handles layout correctly)
    auto S = torch::mm(P, C.t());  // This just works!

    // Step 3: Form expanded distance with custom kernel
    dim3 block(16, 16);
    dim3 grid((M + 15) / 16, (N + 15) / 16);
    form_expanded_rowmajor_float<<<grid, block>>>(
        S.data_ptr<float>(), P2.data_ptr<float>(), C2.data_ptr<float>(),
        D.data_ptr<float>(), N, M);

    cudaDeviceSynchronize();
    return D;
}

torch::Tensor pairwise_euclidean_double(torch::Tensor P, torch::Tensor C) {
    CHECK_INPUT(P);
    CHECK_INPUT(C);
    TORCH_CHECK(P.dtype() == torch::kFloat64, "P must be float64");
    TORCH_CHECK(C.dtype() == torch::kFloat64, "C must be float64");
    TORCH_CHECK(P.dim() == 2 && C.dim() == 2, "Inputs must be 2D");
    
    const int N = P.size(0);
    const int M = C.size(0);
    const int dim = P.size(1);
    TORCH_CHECK(C.size(1) == dim, "Dimension mismatch");

    auto P2 = torch::empty({N}, P.options());
    auto C2 = torch::empty({M}, C.options());
    auto D = torch::empty({N, M}, P.options());

    const int threads = 256;
    compute_norms_rowmajor_double<<<(N + threads - 1) / threads, threads>>>(
        P.data_ptr<double>(), P2.data_ptr<double>(), N, dim);
    compute_norms_rowmajor_double<<<(M + threads - 1) / threads, threads>>>(
        C.data_ptr<double>(), C2.data_ptr<double>(), M, dim);

    auto S = torch::mm(P, C.t());

    dim3 block(16, 16);
    dim3 grid((M + 15) / 16, (N + 15) / 16);
    form_expanded_rowmajor_double<<<grid, block>>>(
        S.data_ptr<double>(), P2.data_ptr<double>(), C2.data_ptr<double>(),
        D.data_ptr<double>(), N, M);

    cudaDeviceSynchronize();
    return D;
}
