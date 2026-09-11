/*
 * PyTorch bindings for advanced mixed-precision kernels, for test only
 */
#include "euclidean_kernels_advanced.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Advanced mixed-precision pairwise Euclidean distance";
    
    // Mode: FP64 GEMM + FP16 storage/fallback
    m.def("pairwise_euclidean_fp64_fp16", &pairwise_euclidean_fp64_fp16,
          "Double-precision GEMM with FP16 storage and fallback",
          py::arg("P"), py::arg("C"), py::arg("kappa") = 5.0f);
    
    // Mode: FP64 GEMM + BF16 storage/fallback
    m.def("pairwise_euclidean_fp64_bf16", &pairwise_euclidean_fp64_bf16,
          "Double-precision GEMM with BF16 storage and fallback",
          py::arg("P"), py::arg("C"), py::arg("kappa") = 5.0f);
    
    // Mode: FP64 with TF32 GEMM
    m.def("pairwise_euclidean_fp64_tf32", &pairwise_euclidean_fp64_tf32,
          "FP64 reference with TF32 GEMM acceleration",
          py::arg("P"), py::arg("C"), py::arg("kappa") = 5.0f);
    
    // Mode: FP64 with FP32 GEMM correction
    m.def("pairwise_euclidean_fp64_fp32_gemm", &pairwise_euclidean_fp64_fp32_gemm,
          "FP64 with FP32 GEMM and adaptive correction",
          py::arg("P"), py::arg("C"), py::arg("kappa") = 5.0f);
    
    // Utility
    m.def("get_last_fallback_count", &get_last_fallback_count,
          "Get count of fallback corrections from last run");
}