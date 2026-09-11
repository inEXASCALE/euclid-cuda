#pragma once
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// ============================================================================
// PHASE 2: SHARED MEMORY TILING - ALL COMBINATIONS
// ============================================================================

// FP16 primary + FP64 fallback
torch::Tensor pairwise_euclidean_tiled_fp16_fp64(
    torch::Tensor P, torch::Tensor C, float kappa, int tile_size = 32);

std::tuple<torch::Tensor, int, std::vector<int>> 
pairwise_euclidean_tiled_fp16_fp64_with_stats(
    torch::Tensor P, torch::Tensor C, float kappa, int tile_size = 32);

// FP16 primary + FP32 fallback
torch::Tensor pairwise_euclidean_tiled_fp16_fp32(
    torch::Tensor P, torch::Tensor C, float kappa, int tile_size = 32);

std::tuple<torch::Tensor, int, std::vector<int>> 
pairwise_euclidean_tiled_fp16_fp32_with_stats(
    torch::Tensor P, torch::Tensor C, float kappa, int tile_size = 32);

// BF16 primary + FP64 fallback
torch::Tensor pairwise_euclidean_tiled_bf16_fp64(
    torch::Tensor P, torch::Tensor C, float kappa, int tile_size = 32);

std::tuple<torch::Tensor, int, std::vector<int>> 
pairwise_euclidean_tiled_bf16_fp64_with_stats(
    torch::Tensor P, torch::Tensor C, float kappa, int tile_size = 32);

// BF16 primary + FP32 fallback
torch::Tensor pairwise_euclidean_tiled_bf16_fp32(
    torch::Tensor P, torch::Tensor C, float kappa, int tile_size = 32);

std::tuple<torch::Tensor, int, std::vector<int>> 
pairwise_euclidean_tiled_bf16_fp32_with_stats(
    torch::Tensor P, torch::Tensor C, float kappa, int tile_size = 32);

// FP32 primary + FP64 fallback
torch::Tensor pairwise_euclidean_tiled_fp32_fp64(
    torch::Tensor P, torch::Tensor C, float kappa, int tile_size = 64);

std::tuple<torch::Tensor, int, std::vector<int>> 
pairwise_euclidean_tiled_fp32_fp64_with_stats(
    torch::Tensor P, torch::Tensor C, float kappa, int tile_size = 64);

// FP32 primary + FP32 fallback (direct formula only)
torch::Tensor pairwise_euclidean_tiled_fp32_fp32(
    torch::Tensor P, torch::Tensor C, float kappa, int tile_size = 64);

std::tuple<torch::Tensor, int, std::vector<int>> 
pairwise_euclidean_tiled_fp32_fp32_with_stats(
    torch::Tensor P, torch::Tensor C, float kappa, int tile_size = 64);

// ============================================================================
// PHASE 3: TENSOR CORE + ASYNC PIPELINE
// ============================================================================

// FP16 Tensor Core + FP64 fallback
torch::Tensor pairwise_euclidean_tensorcore_fp16_fp64(
    torch::Tensor P, torch::Tensor C, float kappa, int tile_size = 64);

// FP16 Tensor Core + FP32 fallback
torch::Tensor pairwise_euclidean_tensorcore_fp16_fp32(
    torch::Tensor P, torch::Tensor C, float kappa, int tile_size = 64);

// BF16 Tensor Core + FP64 fallback (Ampere+)
torch::Tensor pairwise_euclidean_tensorcore_bf16_fp64(
    torch::Tensor P, torch::Tensor C, float kappa, int tile_size = 64);

// BF16 Tensor Core + FP32 fallback (Ampere+)
torch::Tensor pairwise_euclidean_tensorcore_bf16_fp32(
    torch::Tensor P, torch::Tensor C, float kappa, int tile_size = 64);

// TF32 Tensor Core + FP64 fallback (Ampere+)
torch::Tensor pairwise_euclidean_tensorcore_tf32_fp64(
    torch::Tensor P, torch::Tensor C, float kappa, int tile_size = 64);