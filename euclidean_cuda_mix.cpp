/*
 * PyTorch bindings for mixed-precision kernels with various fallback strategies
 */
#include "euclidean_kernels_mix.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Mixed-precision Euclidean distance with adaptive fallback strategies";
    
    // ========================================================================
    // EXISTING: FP64 Fallback Variants
    // ========================================================================
    
    m.def("pairwise_euclidean_fp16_fp64", &pairwise_euclidean_fp16_fp64,
          "FP16 GEMM with FP64 fallback for unstable entries",
          py::arg("P"), py::arg("C"), py::arg("kappa") = 5.0f);
    
    m.def("pairwise_euclidean_bf16_fp64", &pairwise_euclidean_bf16_fp64,
          "BF16 GEMM with FP64 fallback for unstable entries",
          py::arg("P"), py::arg("C"), py::arg("kappa") = 5.0f);
    
    m.def("pairwise_euclidean_tf32_fp64", &pairwise_euclidean_tf32_fp64,
          "TF32 GEMM with FP64 fallback for unstable entries",
          py::arg("P"), py::arg("C"), py::arg("kappa") = 5.0f);

    m.def("pairwise_euclidean_fp32_fp64", &pairwise_euclidean_fp32_fp64,
          "FP32 GEMM with FP64 fallback for unstable entries",
          py::arg("P"), py::arg("C"), py::arg("kappa") = 5.0f);

    m.def("pairwise_euclidean_fp16_fp64_with_stats", &pairwise_euclidean_fp16_fp64_with_stats,
          "FP16+FP64 with fallback statistics",
          py::arg("P"), py::arg("C"), py::arg("kappa") = 5.0f);
    
    m.def("pairwise_euclidean_bf16_fp64_with_stats", &pairwise_euclidean_bf16_fp64_with_stats,
          "BF16+FP64 with fallback statistics",
          py::arg("P"), py::arg("C"), py::arg("kappa") = 5.0f);
    
    m.def("pairwise_euclidean_tf32_fp64_with_stats", &pairwise_euclidean_tf32_fp64_with_stats,
          "TF32+FP64 with fallback statistics",
          py::arg("P"), py::arg("C"), py::arg("kappa") = 5.0f);

    m.def("pairwise_euclidean_fp32_fp64_with_stats", &pairwise_euclidean_fp32_fp64_with_stats,
          "FP32+FP64 with fallback statistics",
          py::arg("P"), py::arg("C"), py::arg("kappa") = 5.0f);

    // ========================================================================
    // NEW: Direct Formula Fallback Variants
    // ========================================================================
    
    m.def("pairwise_euclidean_fp16_fp16", &pairwise_euclidean_fp16_fp16,
          "FP16 GEMM with FP16 direct formula fallback",
          py::arg("P"), py::arg("C"), py::arg("kappa") = 5.0f);
    
    m.def("pairwise_euclidean_fp16_fp32", &pairwise_euclidean_fp16_fp32,
          "FP16 GEMM with FP32 direct formula fallback",
          py::arg("P"), py::arg("C"), py::arg("kappa") = 5.0f);
    
    m.def("pairwise_euclidean_bf16_bf16", &pairwise_euclidean_bf16_bf16,
          "BF16 GEMM with BF16 direct formula fallback",
          py::arg("P"), py::arg("C"), py::arg("kappa") = 5.0f);
    
    m.def("pairwise_euclidean_bf16_fp32", &pairwise_euclidean_bf16_fp32,
          "BF16 GEMM with FP32 direct formula fallback",
          py::arg("P"), py::arg("C"), py::arg("kappa") = 5.0f);
    
    m.def("pairwise_euclidean_fp32_fp32", &pairwise_euclidean_fp32_fp32,
          "FP32 GEMM with FP32 direct formula fallback",
          py::arg("P"), py::arg("C"), py::arg("kappa") = 5.0f);

    // With statistics
    m.def("pairwise_euclidean_fp16_fp16_with_stats", &pairwise_euclidean_fp16_fp16_with_stats,
          "FP16+FP16 direct fallback with statistics",
          py::arg("P"), py::arg("C"), py::arg("kappa") = 5.0f);
    
    m.def("pairwise_euclidean_fp16_fp32_with_stats", &pairwise_euclidean_fp16_fp32_with_stats,
          "FP16+FP32 direct fallback with statistics",
          py::arg("P"), py::arg("C"), py::arg("kappa") = 5.0f);
    
    m.def("pairwise_euclidean_bf16_bf16_with_stats", &pairwise_euclidean_bf16_bf16_with_stats,
          "BF16+BF16 direct fallback with statistics",
          py::arg("P"), py::arg("C"), py::arg("kappa") = 5.0f);
    
    m.def("pairwise_euclidean_bf16_fp32_with_stats", &pairwise_euclidean_bf16_fp32_with_stats,
          "BF16+FP32 direct fallback with statistics",
          py::arg("P"), py::arg("C"), py::arg("kappa") = 5.0f);
    
    m.def("pairwise_euclidean_fp32_fp32_with_stats", &pairwise_euclidean_fp32_fp32_with_stats,
          "FP32+FP32 direct fallback with statistics",
          py::arg("P"), py::arg("C"), py::arg("kappa") = 5.0f);

    // ========================================================================
    // Utilities
    // ========================================================================
    
    m.def("get_fallback_count_low", &get_fallback_count_low,
          "Get fallback count from last computation");
}