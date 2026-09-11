/*
 * Advanced Mixed-Precision Euclidean Distance - Header
 * FIXED: Use c10::Half and c10::BFloat16 types
 */
#pragma once
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/core/ScalarType.h>
#include <cublas_v2.h> 

// Use c10 types instead of CUDA native types in kernel signatures
__global__ void compute_norms_fp16(
    const c10::Half* X, float* X2, int N, int dim);

__global__ void compute_norms_bf16(
    const c10::BFloat16* X, float* X2, int N, int dim);

__global__ void compute_norms_rowmajor_float(
    const float* X, float* X2, int N, int dim);

__global__ void compute_norms_rowmajor_double(
    const double* X, double* X2, int N, int dim);

__global__ void form_expanded_rowmajor_double_to_float(
    const double* S, const float* P2, const float* C2,
    float* D, int N, int M);

__global__ void form_mixed_and_list_double(
    const double* S, const float* P2, const float* C2,
    float* D, uint32_t* idx_list, uint32_t* list_count,
    int N, int M, float gamma_n, float kappa);

__global__ void form_mixed_and_list_double_output(
    const double* S, const double* P2, const double* C2,
    double* D, uint32_t* idx_list, uint32_t* list_count,
    int N, int M, double gamma_n, double kappa);

__global__ void direct_on_list_fp16(
    const c10::Half* P, const c10::Half* C, float* D,
    const uint32_t* idx_list, const uint32_t* list_count,
    int dim, int N, int M);

__global__ void direct_on_list_bf16(
    const c10::BFloat16* P, const c10::BFloat16* C, float* D,
    const uint32_t* idx_list, const uint32_t* list_count,
    int dim, int N, int M);

__global__ void direct_on_list_fp64(
    const double* P, const double* C, double* D,
    const uint32_t* idx_list, const uint32_t* list_count,
    int dim, int N, int M);

// Host functions
torch::Tensor pairwise_euclidean_fp64_fp16(
    torch::Tensor P, torch::Tensor C, float kappa);

torch::Tensor pairwise_euclidean_fp64_bf16(
    torch::Tensor P, torch::Tensor C, float kappa);

torch::Tensor pairwise_euclidean_fp64_tf32(
    torch::Tensor P, torch::Tensor C, float kappa);

torch::Tensor pairwise_euclidean_fp64_fp32_gemm(
    torch::Tensor P, torch::Tensor C, float kappa);

int get_last_fallback_count(torch::Tensor list_count_tensor);


std::tuple<torch::Tensor, int> pairwise_euclidean_fp64_fp16_with_stats(torch::Tensor P, torch::Tensor C, float kappa);
std::tuple<torch::Tensor, int> pairwise_euclidean_fp64_bf16_with_stats(torch::Tensor P, torch::Tensor C, float kappa);
std::tuple<torch::Tensor, int> pairwise_euclidean_fp64_tf32_with_stats(torch::Tensor P, torch::Tensor C, float kappa);
std::tuple<torch::Tensor, int> pairwise_euclidean_fp64_fp32_gemm_with_stats(torch::Tensor P, torch::Tensor C, float kappa);