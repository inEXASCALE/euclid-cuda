/*
 * PyTorch CUDA Extension: Fast Euclidean Distance
 * Header file
 */
#pragma once
#include <torch/extension.h>
#include <cuda_runtime.h>

// Row-major kernels
__global__ void compute_norms_rowmajor_float(
    const float* X, float* X2, int N, int dim);

__global__ void compute_norms_rowmajor_double(
    const double* X, double* X2, int N, int dim);

__global__ void form_expanded_rowmajor_float(
    const float* S, const float* P2, const float* C2,
    float* D, int N, int M);

__global__ void form_expanded_rowmajor_double(
    const double* S, const double* P2, const double* C2,
    double* D, int N, int M);

__global__ void form_mixed_and_list_rowmajor(
    const float* S, const float* P2, const float* C2,
    float* D, uint32_t* idx_list, uint32_t* list_count,
    int N, int M, float gamma_n, float kappa);


// Host functions
torch::Tensor pairwise_euclidean_single(torch::Tensor P, torch::Tensor C);
torch::Tensor pairwise_euclidean_double(torch::Tensor P, torch::Tensor C);