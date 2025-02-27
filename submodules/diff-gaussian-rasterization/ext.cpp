/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <torch/extension.h>
#include "rasterize_points.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
  m.def("batch_rasterize_gaussians1", &BatchRasterizeGaussiansCUDA1);
  m.def("batch_rasterize_gaussians2", &BatchRasterizeGaussiansCUDA2);
  m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
  m.def("batch_rasterize_gaussians_backward", &BatchRasterizeGaussiansBackwardCUDA);
  m.def("mark_visible", &markVisible);
  m.def("compute_face_tbn", &ComputeFaceTBNCUDA);
  m.def("matrix_to_quaternion", &MatrixToQuaternionCUDA);
  m.def("matrix_to_quaternion_backward", &MatrixToQuaternionBackwardCUDA);
  m.def("quaternion_multiply", &QuaternionMultiplyCUDA);
  m.def("quaternion_multiply_backward", &QuaternionMultiplyBackwardCUDA);
  m.def("fast_forward", &FastForwardCUDA);
  m.def("mesh_binding", &MeshBindingCUDA);
  m.def("mesh_binding_backward", &MeshBindingBackwardCUDA);
  m.def("linear_blending", &LinearBlendingCUDA);
  m.def("linear_blending_backward", &LinearBlendingBackwardCUDA);
}