#pragma once
#include <torch/extension.h>

// FP32 variants
torch::Tensor update_centers_fp32(
    torch::Tensor X,
    torch::Tensor labels,
    int n_clusters
);

torch::Tensor update_centers_fp32_with_reinit(
    torch::Tensor X,
    torch::Tensor labels,
    int n_clusters
);

// FP64 variants
torch::Tensor update_centers_fp64(
    torch::Tensor X,
    torch::Tensor labels,
    int n_clusters
);

torch::Tensor update_centers_fp64_with_reinit(
    torch::Tensor X,
    torch::Tensor labels,
    int n_clusters
);