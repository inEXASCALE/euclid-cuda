#include <torch/extension.h>
#include "euclidean_kernels_tiled.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Tiled mixed-precision Euclidean distance kernels (all combinations)";
    
    // ========================================================================
    // PHASE 2: SHARED MEMORY TILING
    // ========================================================================
    
    // FP16 variants
    m.def("pairwise_euclidean_tiled_fp16_fp64", &pairwise_euclidean_tiled_fp16_fp64,
          py::arg("P"), py::arg("C"), py::arg("kappa"), py::arg("tile_size") = 32);
    m.def("pairwise_euclidean_tiled_fp16_fp32", &pairwise_euclidean_tiled_fp16_fp32,
          py::arg("P"), py::arg("C"), py::arg("kappa"), py::arg("tile_size") = 32);
    m.def("pairwise_euclidean_tiled_fp16_fp64_with_stats", 
          &pairwise_euclidean_tiled_fp16_fp64_with_stats,
          py::arg("P"), py::arg("C"), py::arg("kappa"), py::arg("tile_size") = 32);
    m.def("pairwise_euclidean_tiled_fp16_fp32_with_stats", 
          &pairwise_euclidean_tiled_fp16_fp32_with_stats,
          py::arg("P"), py::arg("C"), py::arg("kappa"), py::arg("tile_size") = 32);
    
    // BF16 variants
    m.def("pairwise_euclidean_tiled_bf16_fp64", &pairwise_euclidean_tiled_bf16_fp64,
          py::arg("P"), py::arg("C"), py::arg("kappa"), py::arg("tile_size") = 32);
    m.def("pairwise_euclidean_tiled_bf16_fp32", &pairwise_euclidean_tiled_bf16_fp32,
          py::arg("P"), py::arg("C"), py::arg("kappa"), py::arg("tile_size") = 32);
    m.def("pairwise_euclidean_tiled_bf16_fp64_with_stats", 
          &pairwise_euclidean_tiled_bf16_fp64_with_stats,
          py::arg("P"), py::arg("C"), py::arg("kappa"), py::arg("tile_size") = 32);
    m.def("pairwise_euclidean_tiled_bf16_fp32_with_stats", 
          &pairwise_euclidean_tiled_bf16_fp32_with_stats,
          py::arg("P"), py::arg("C"), py::arg("kappa"), py::arg("tile_size") = 32);
    
    // FP32 variants
    m.def("pairwise_euclidean_tiled_fp32_fp64", &pairwise_euclidean_tiled_fp32_fp64,
          py::arg("P"), py::arg("C"), py::arg("kappa"), py::arg("tile_size") = 64);
    m.def("pairwise_euclidean_tiled_fp32_fp32", &pairwise_euclidean_tiled_fp32_fp32,
          py::arg("P"), py::arg("C"), py::arg("kappa"), py::arg("tile_size") = 64);
    m.def("pairwise_euclidean_tiled_fp32_fp64_with_stats", 
          &pairwise_euclidean_tiled_fp32_fp64_with_stats,
          py::arg("P"), py::arg("C"), py::arg("kappa"), py::arg("tile_size") = 64);
    m.def("pairwise_euclidean_tiled_fp32_fp32_with_stats", 
          &pairwise_euclidean_tiled_fp32_fp32_with_stats,
          py::arg("P"), py::arg("C"), py::arg("kappa"), py::arg("tile_size") = 64);
    
    // ========================================================================
    // PHASE 3: TENSOR CORE
    // ========================================================================
    
    m.def("pairwise_euclidean_tensorcore_fp16_fp64", 
          &pairwise_euclidean_tensorcore_fp16_fp64,
          py::arg("P"), py::arg("C"), py::arg("kappa"), py::arg("tile_size") = 64);
    m.def("pairwise_euclidean_tensorcore_fp16_fp32", 
          &pairwise_euclidean_tensorcore_fp16_fp32,
          py::arg("P"), py::arg("C"), py::arg("kappa"), py::arg("tile_size") = 64);
    m.def("pairwise_euclidean_tensorcore_bf16_fp64", 
          &pairwise_euclidean_tensorcore_bf16_fp64,
          py::arg("P"), py::arg("C"), py::arg("kappa"), py::arg("tile_size") = 64);
    m.def("pairwise_euclidean_tensorcore_bf16_fp32", 
          &pairwise_euclidean_tensorcore_bf16_fp32,
          py::arg("P"), py::arg("C"), py::arg("kappa"), py::arg("tile_size") = 64);
    // m.def("pairwise_euclidean_tensorcore_tf32_fp64", 
    //      &pairwise_euclidean_tensorcore_tf32_fp64,
    //      py::arg("P"), py::arg("C"), py::arg("kappa"), py::arg("tile_size") = 64);
}