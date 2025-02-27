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

#pragma once
#include <cstdio>
#include <tuple>
#include <string>

#include <torch/extension.h>
#include <cuda_runtime_api.h>


int RasterizeGaussiansCUDA(
	const int64_t stream,
	const torch::Tensor& background,
	const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& target_image,
	torch::Tensor& geomBuffer,
	torch::Tensor& binningBuffer,
	torch::Tensor& imgBuffer,
	torch::Tensor& out_color,
	torch::Tensor& out_alpha,
	torch::Tensor& est_color,
	torch::Tensor& est_weight,
	torch::Tensor& radii,
	const bool prefiltered,
	const bool debug
);

void BatchRasterizeGaussiansCUDA1(
	const int64_t stream,
	const int batch_index,
	torch::Tensor& num_rendered,
	const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	torch::Tensor& geomBuffer,
	torch::Tensor& radii,
	const bool prefiltered,
	const bool debug
);

void BatchRasterizeGaussiansCUDA2(
	const int64_t stream,
	const int batch_index,
	const int P, 
	const torch::Tensor& num_rendered,
	const torch::Tensor& background,
    const torch::Tensor& colors,
    const int image_height,
    const int image_width,
	const torch::Tensor& target_image,
	torch::Tensor& geomBuffer,
	torch::Tensor& binningBuffer,
	torch::Tensor& imgBuffer,
	torch::Tensor& out_color,
	torch::Tensor& out_alpha,
	torch::Tensor& est_color,
	torch::Tensor& est_weight,
	torch::Tensor& radii,
	const bool debug
);

void RasterizeGaussiansBackwardCUDA(
	const int64_t stream,
 	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
    const torch::Tensor& colors,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const torch::Tensor& dL_dout_color,
	const torch::Tensor& dL_dout_alpha,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const torch::Tensor& out_alpha,
	torch::Tensor& dL_dmeans3D,
	torch::Tensor& dL_dmeans2D,
	torch::Tensor& dL_dcolors,
	torch::Tensor& dL_dconic,
	torch::Tensor& dL_dopacity,
	torch::Tensor& dL_dcov3D,
	torch::Tensor& dL_dsh,
	torch::Tensor& dL_dscales,
	torch::Tensor& dL_drotations,
	const bool debug
);

void BatchRasterizeGaussiansBackwardCUDA(
	const int64_t stream,
	const int batch_index,
 	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
    const torch::Tensor& colors,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const torch::Tensor& dL_dout_color,
	const torch::Tensor& dL_dout_alpha,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const torch::Tensor& out_alpha,
	torch::Tensor& dL_dmeans3D,
	torch::Tensor& dL_dmeans2D,
	torch::Tensor& dL_dcolors,
	torch::Tensor& dL_dconic,
	torch::Tensor& dL_dopacity,
	torch::Tensor& dL_dcov3D,
	torch::Tensor& dL_dsh,
	torch::Tensor& dL_dscales,
	torch::Tensor& dL_drotations,
	const bool debug
);
		
torch::Tensor markVisible(
	torch::Tensor& means3D,
	torch::Tensor& viewmatrix,
	torch::Tensor& projmatrix
);

torch::Tensor ComputeFaceTBNCUDA(const torch::Tensor& face_verts, const torch::Tensor& face_uvs);

torch::Tensor MatrixToQuaternionCUDA(const torch::Tensor& matrix);

torch::Tensor MatrixToQuaternionBackwardCUDA(const torch::Tensor& grad_quaternion, const torch::Tensor& matrix);

torch::Tensor QuaternionMultiplyCUDA(const torch::Tensor& q1, const torch::Tensor& q2);

std::tuple<torch::Tensor, torch::Tensor> QuaternionMultiplyBackwardCUDA(
	const torch::Tensor& grad_product, 
	const torch::Tensor& q1,
	const torch::Tensor& q2
);

void FastForwardCUDA(
	const float weight_threshold,
	const torch::Tensor& est_color,
	const torch::Tensor& est_weight,
	torch::Tensor& initialized,
	torch::Tensor& feature_dc
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> MeshBindingCUDA(
	const torch::Tensor& face_verts,
	const torch::Tensor& face_tbns,
	const torch::Tensor& binding_face_barys,
	const torch::Tensor& binding_face_ids,
	const torch::Tensor& gs_xyzs,
	const torch::Tensor& gs_rots
);

std::tuple<torch::Tensor, torch::Tensor> MeshBindingBackwardCUDA(
	const torch::Tensor& grad_transformed_gs_xyzs,
    const torch::Tensor& grad_transformed_gs_rots,
    const torch::Tensor& binding_rotations
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> LinearBlendingCUDA(
	const torch::Tensor& blending_weights,
	const torch::Tensor& base_xyzs,
	const torch::Tensor& base_rots,
	const torch::Tensor& base_rgbs,
	const torch::Tensor& delta_xyzs,
	const torch::Tensor& delta_rots,
	const torch::Tensor& delta_rgbs
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> LinearBlendingBackwardCUDA(
	const torch::Tensor& blending_weights,
	const torch::Tensor& delta_xyzs,
	const torch::Tensor& delta_rots,
	const torch::Tensor& delta_rgbs,
	const torch::Tensor& grad_blended_xyzs,
	const torch::Tensor& grad_blended_rots,
	const torch::Tensor& grad_blended_rgbs
);