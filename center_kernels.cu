/*
 * KMeans Center Update Kernels
 * FP32 and FP64 implementations
 */
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// ============================================================================
// COUNT POINTS PER CLUSTER
// ============================================================================

__global__ void count_points_per_cluster_kernel(
    const int64_t* __restrict__ labels,
    int* __restrict__ counts,
    const int N,
    const int K
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < N) {
        int k = labels[tid];
        if (k >= 0 && k < K) {  // Bounds check
            atomicAdd(&counts[k], 1);
        }
    }
}

// ============================================================================
// COMPUTE CENTERS - FP32
// ============================================================================

__global__ void compute_centers_fp32_kernel(
    const float* __restrict__ X,
    const int64_t* __restrict__ labels,
    const int* __restrict__ counts,
    float* __restrict__ centers,
    const int N,
    const int D,
    const int K
) {
    int k = blockIdx.y;  // Cluster index
    int d = blockIdx.x * blockDim.x + threadIdx.x;  // Dimension index
    
    if (k >= K || d >= D) return;
    
    int count = counts[k];
    if (count == 0) return;  // Empty cluster, will be handled later
    
    // Accumulate sum in FP32
    float sum = 0.0f;
    for (int n = 0; n < N; ++n) {
        if (labels[n] == k) {
            sum += X[n * D + d];
        }
    }
    
    centers[k * D + d] = sum / static_cast<float>(count);
}

// ============================================================================
// COMPUTE CENTERS - FP64
// ============================================================================

__global__ void compute_centers_fp64_kernel(
    const double* __restrict__ X,
    const int64_t* __restrict__ labels,
    const int* __restrict__ counts,
    double* __restrict__ centers,
    const int N,
    const int D,
    const int K
) {
    int k = blockIdx.y;  // Cluster index
    int d = blockIdx.x * blockDim.x + threadIdx.x;  // Dimension index
    
    if (k >= K || d >= D) return;
    
    int count = counts[k];
    if (count == 0) return;  // Empty cluster
    
    // Accumulate sum in FP64 for maximum precision
    double sum = 0.0;
    for (int n = 0; n < N; ++n) {
        if (labels[n] == k) {
            sum += X[n * D + d];
        }
    }
    
    centers[k * D + d] = sum / static_cast<double>(count);
}

// ============================================================================
// HOST WRAPPER FUNCTIONS
// ============================================================================

torch::Tensor update_centers_fp32(
    torch::Tensor X,           // (N, D) float32
    torch::Tensor labels,      // (N,) int64
    int n_clusters
) {
    CHECK_INPUT(X);
    CHECK_INPUT(labels);
    TORCH_CHECK(X.dtype() == torch::kFloat32, "X must be float32");
    TORCH_CHECK(labels.dtype() == torch::kInt64, "labels must be int64");
    
    const int N = X.size(0);
    const int D = X.size(1);
    const int K = n_clusters;
    
    // Allocate outputs
    auto centers = torch::zeros({K, D}, X.options());
    auto counts = torch::zeros({K}, torch::TensorOptions().dtype(torch::kInt32).device(X.device()));
    
    const int threads = 256;
    
    // Pass 1: Count points per cluster
    int blocks_count = (N + threads - 1) / threads;
    count_points_per_cluster_kernel<<<blocks_count, threads>>>(
        labels.data_ptr<int64_t>(),
        counts.data_ptr<int>(),
        N, K
    );
    
    // Pass 2: Compute centers (parallelize over K x D)
    dim3 block(256);
    dim3 grid((D + 255) / 256, K);
    
    compute_centers_fp32_kernel<<<grid, block>>>(
        X.data_ptr<float>(),
        labels.data_ptr<int64_t>(),
        counts.data_ptr<int>(),
        centers.data_ptr<float>(),
        N, D, K
    );
    
    cudaDeviceSynchronize();
    return centers;
}

torch::Tensor update_centers_fp64(
    torch::Tensor X,           // (N, D) float64
    torch::Tensor labels,      // (N,) int64
    int n_clusters
) {
    CHECK_INPUT(X);
    CHECK_INPUT(labels);
    TORCH_CHECK(X.dtype() == torch::kFloat64, "X must be float64");
    TORCH_CHECK(labels.dtype() == torch::kInt64, "labels must be int64");
    
    const int N = X.size(0);
    const int D = X.size(1);
    const int K = n_clusters;
    
    // Allocate outputs
    auto centers = torch::zeros({K, D}, X.options());
    auto counts = torch::zeros({K}, torch::TensorOptions().dtype(torch::kInt32).device(X.device()));
    
    const int threads = 256;
    
    // Pass 1: Count points per cluster
    int blocks_count = (N + threads - 1) / threads;
    count_points_per_cluster_kernel<<<blocks_count, threads>>>(
        labels.data_ptr<int64_t>(),
        counts.data_ptr<int>(),
        N, K
    );
    
    // Pass 2: Compute centers (parallelize over K x D)
    dim3 block(256);
    dim3 grid((D + 255) / 256, K);
    
    compute_centers_fp64_kernel<<<grid, block>>>(
        X.data_ptr<double>(),
        labels.data_ptr<int64_t>(),
        counts.data_ptr<int>(),
        centers.data_ptr<double>(),
        N, D, K
    );
    
    cudaDeviceSynchronize();
    return centers;
}

// ============================================================================
// WITH EMPTY CLUSTER REINITIALIZATION
// ============================================================================

torch::Tensor update_centers_fp32_with_reinit(
    torch::Tensor X,
    torch::Tensor labels,
    int n_clusters
) {
    auto centers = update_centers_fp32(X, labels, n_clusters);
    
    // Check for empty clusters
    auto counts = torch::bincount(labels, torch::Tensor(), n_clusters);
    auto empty_mask = (counts == 0);
    int n_empty = empty_mask.sum().item<int>();
    
    if (n_empty > 0) {
        // Reinitialize empty clusters with random points
        auto empty_indices = torch::nonzero(empty_mask).squeeze(1);
        auto random_points = torch::randint(0, X.size(0), {n_empty}, 
            torch::TensorOptions().dtype(torch::kLong).device(X.device()));
        
        for (int i = 0; i < n_empty; ++i) {
            int k = empty_indices[i].item<int>();
            int rand_idx = random_points[i].item<int>();
            centers[k] = X[rand_idx];
        }
    }
    
    return centers;
}

torch::Tensor update_centers_fp64_with_reinit(
    torch::Tensor X,
    torch::Tensor labels,
    int n_clusters
) {
    auto centers = update_centers_fp64(X, labels, n_clusters);
    
    // Check for empty clusters
    auto counts = torch::bincount(labels, torch::Tensor(), n_clusters);
    auto empty_mask = (counts == 0);
    int n_empty = empty_mask.sum().item<int>();
    
    if (n_empty > 0) {
        auto empty_indices = torch::nonzero(empty_mask).squeeze(1);
        auto random_points = torch::randint(0, X.size(0), {n_empty}, 
            torch::TensorOptions().dtype(torch::kLong).device(X.device()));
        
        for (int i = 0; i < n_empty; ++i) {
            int k = empty_indices[i].item<int>();
            int rand_idx = random_points[i].item<int>();
            centers[k] = X[rand_idx];
        }
    }
    
    return centers;
}

// ============================================================================
// PYBIND11 BINDINGS
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "KMeans center update kernels (FP32 and FP64)";
    
    m.def("update_centers_fp32", &update_centers_fp32,
          "Update KMeans centers in FP32",
          py::arg("X"), py::arg("labels"), py::arg("n_clusters"));
    
    m.def("update_centers_fp64", &update_centers_fp64,
          "Update KMeans centers in FP64",
          py::arg("X"), py::arg("labels"), py::arg("n_clusters"));
    
    m.def("update_centers_fp32_with_reinit", &update_centers_fp32_with_reinit,
          "Update KMeans centers in FP32 with empty cluster reinitialization",
          py::arg("X"), py::arg("labels"), py::arg("n_clusters"));
    
    m.def("update_centers_fp64_with_reinit", &update_centers_fp64_with_reinit,
          "Update KMeans centers in FP64 with empty cluster reinitialization",
          py::arg("X"), py::arg("labels"), py::arg("n_clusters"));
}