/*
 * Uniform Precision Euclidean Distance Kernels
 * Pure FP16, BF16, TF32 implementations (no fallback)
 */
#pragma once
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/core/ScalarType.h>

// Uniform FP16
torch::Tensor pairwise_euclidean_uniform_fp16(
    torch::Tensor P, torch::Tensor C);

// Uniform BF16
torch::Tensor pairwise_euclidean_uniform_bf16(
    torch::Tensor P, torch::Tensor C);

// Uniform TF32 (uses FP32 input but TF32 GEMM)
torch::Tensor pairwise_euclidean_uniform_tf32(
    torch::Tensor P, torch::Tensor C);


// ============================================================================
// DIRECT FORMULA KERNELS - FP16
// More stable for near points, no expand/GEMM needed
// ============================================================================

// Uniform FP16 (direct)
torch::Tensor pairwise_euclidean_uniform_fp16_direct(
    torch::Tensor P, torch::Tensor C);

// Uniform BF16 (direct)
torch::Tensor pairwise_euclidean_uniform_bf16_direct(
    torch::Tensor P, torch::Tensor C);

// Uniform TF32 (direct)
torch::Tensor pairwise_euclidean_uniform_tf32_direct(
    torch::Tensor P, torch::Tensor C);


// Uniform FP32 (direct)
torch::Tensor pairwise_euclidean_uniform_fp32_direct(
    torch::Tensor P, torch::Tensor C);

// Uniform FP64 (direct)
torch::Tensor pairwise_euclidean_uniform_fp64_direct(
    torch::Tensor P, torch::Tensor C);



// Uniform FP32 (direct)
torch::Tensor pairwise_euclidean_uniform_fp32_direct(
    torch::Tensor P, torch::Tensor C);

// Uniform FP64 (direct)
torch::Tensor pairwise_euclidean_uniform_fp64_direct(
    torch::Tensor P, torch::Tensor C);