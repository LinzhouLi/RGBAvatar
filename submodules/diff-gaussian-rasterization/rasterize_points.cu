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

#include <math.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <memory>
#include <fstream>
#include <string>
#include <functional>

#include <cuda_runtime_api.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "cuda_utils/utils.h"
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"


std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t, const at::cuda::CUDAStream& stream) {
    auto lambda = [&t, &stream](size_t N) {
		at::cuda::CUDAStreamGuard guard(stream);
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

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
) {
	if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
		AT_ERROR("means3D must have dimensions (num_points, 3)");
	}
	
	const int P = means3D.size(0);
	const int H = image_height;
	const int W = image_width;

	cudaStream_t raw_stream = reinterpret_cast<cudaStream_t>(stream);
	at::cuda::CUDAStream at_stream = at::cuda::getStreamFromExternal(raw_stream, means3D.device().index());

	std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer, at_stream);
	std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer, at_stream);
	std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer, at_stream);
	
	int rendered = 0;
	if(P != 0) {
		int M = 0;
		if(sh.size(0) != 0) M = sh.size(1); 

		rendered = CudaRasterizer::Rasterizer::forward(
			raw_stream,
			geomFunc,
			binningFunc,
			imgFunc,
			P, degree, M,
			background.contiguous().data_ptr<float>(),
			W, H,
			means3D.contiguous().data_ptr<float>(),
			sh.contiguous().data_ptr<float>(),
			colors.contiguous().data_ptr<float>(), 
			opacity.contiguous().data_ptr<float>(), 
			scales.contiguous().data_ptr<float>(),
			scale_modifier,
			rotations.contiguous().data_ptr<float>(),
			cov3D_precomp.contiguous().data_ptr<float>(), 
			viewmatrix.contiguous().data_ptr<float>(), 
			projmatrix.contiguous().data_ptr<float>(),
			campos.contiguous().data_ptr<float>(),
			tan_fovx,
			tan_fovy,
			prefiltered,
			out_color.contiguous().data_ptr<float>(),
			out_alpha.contiguous().data_ptr<float>(),
			est_color.contiguous().data_ptr<float>(),
			est_weight.contiguous().data_ptr<float>(),
			radii.contiguous().data_ptr<int>(),
			target_image.contiguous().data_ptr<float>(),
			debug
		);
	}
	return rendered;
}

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
) {
	if (means3D.ndimension() != 3 || means3D.size(2) != 3) {
		AT_ERROR("means3D must have dimensions (batch_size, num_points, 3)");
	}
	
	const int P = means3D.size(1); // caution: with batch dim
	int M = sh.size(0) == 0 ? 0 : sh.size(2);
	
	cudaStream_t raw_stream = reinterpret_cast<cudaStream_t>(stream);
	at::cuda::CUDAStream at_stream = at::cuda::getStreamFromExternal(raw_stream, geomBuffer.device().index());
	std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer, at_stream);

	int* num_rendered_ptr = num_rendered.data_ptr<int>() + batch_index;
	float* means3D_ptr = means3D.data_ptr<float>() + batch_index * P * 3;
	float* sh_ptr = sh.data_ptr<float>() + batch_index * P * 3; // TODO: fix hard code
	float* colors_ptr = nullptr;
	float* opacity_ptr = opacity.data_ptr<float>() + batch_index * P * 1;
	float* scales_ptr = scales.data_ptr<float>() + batch_index * P * 3;
	float* rotations_ptr = rotations.data_ptr<float>() + batch_index * P * 4;
	float* cov3D_precomp_ptr = nullptr;
	int* radii_ptr = radii.data_ptr<int>() + batch_index * P * 1;

	// cudaMemsetAsync(radii_ptr, 0, P * 1 * sizeof(int), raw_stream);
	
	if (P > 0) {
		CudaRasterizer::Rasterizer::forward1(
			raw_stream,
			num_rendered_ptr,
			geomFunc,
			P, degree, M,
			image_width, image_height,
			means3D_ptr,
			sh_ptr,
			colors_ptr, 
			opacity_ptr, 
			scales_ptr,
			scale_modifier,
			rotations_ptr,
			cov3D_precomp_ptr, 
			viewmatrix.contiguous().data_ptr<float>(), 
			projmatrix.contiguous().data_ptr<float>(),
			campos.contiguous().data_ptr<float>(),
			tan_fovx,
			tan_fovy,
			prefiltered,
			radii_ptr,
			debug
		);
	}
	return;
}

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
) {
	const int H = image_height;
	const int W = image_width;

	cudaStream_t raw_stream = reinterpret_cast<cudaStream_t>(stream);
	at::cuda::CUDAStream at_stream = at::cuda::getStreamFromExternal(raw_stream, binningBuffer.device().index());
	std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer, at_stream);
	std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer, at_stream);

	int* num_rendered_ptr = num_rendered.data_ptr<int>() + batch_index;
	float* background_ptr = background.data_ptr<float>() + batch_index * 3;
	float* colors_ptr = nullptr;
	float* out_color_ptr = out_color.data_ptr<float>() + batch_index * 3 * H * W;
	float* out_alpha_ptr = out_alpha.data_ptr<float>() + batch_index * 1 * H * W;
	float* est_color_ptr = est_color.data_ptr<float>() + batch_index * 3 * P;
	float* est_weight_ptr = est_weight.data_ptr<float>() + batch_index * 1 * P;
	int* radii_ptr = radii.data_ptr<int>() + batch_index * P * 1;
	float* target_image_ptr = target_image.data_ptr<float>();
	if (target_image_ptr != nullptr) target_image_ptr += batch_index * 4 * H * W;

	// cudaMemsetAsync(out_color_ptr, 0, 3 * H * W * sizeof(float), raw_stream);
	// cudaMemsetAsync(out_alpha_ptr, 0, 1 * H * W * sizeof(float), raw_stream);
	// cudaMemsetAsync(est_color_ptr, 0, 3 * P * sizeof(float), raw_stream);
	// cudaMemsetAsync(est_weight_ptr, 0, 1 * P * sizeof(float), raw_stream);
	
	if (P != 0) {
		CudaRasterizer::Rasterizer::forward2(
			raw_stream,
			reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
			binningFunc,
			imgFunc,
			P, 
			num_rendered_ptr,
			background_ptr,
			W, H,
			colors_ptr, 
			out_color_ptr,
			out_alpha_ptr,
			est_color_ptr,
			est_weight_ptr,
			radii_ptr,
			target_image_ptr,
			debug
		);
	}
	return;
}

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
) {
	const int P = means3D.size(0);
	const int H = dL_dout_color.size(1);
	const int W = dL_dout_color.size(2);
	const int M = sh.size(0) == 0 ? 0 : sh.size(1);
	
	if(P != 0) {
		CudaRasterizer::Rasterizer::backward(
			reinterpret_cast<cudaStream_t>(stream),
			P, degree, M, R,
			background.contiguous().data_ptr<float>(),
			W, H, 
			means3D.contiguous().data_ptr<float>(),
			sh.contiguous().data_ptr<float>(),
			colors.contiguous().data_ptr<float>(),
			scales.data_ptr<float>(),
			scale_modifier,
			rotations.data_ptr<float>(),
			cov3D_precomp.contiguous().data_ptr<float>(),
			viewmatrix.contiguous().data_ptr<float>(),
			projmatrix.contiguous().data_ptr<float>(),
			campos.contiguous().data_ptr<float>(),
			tan_fovx,
			tan_fovy,
			radii.contiguous().data_ptr<int>(),
			reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
			reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
			reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
			out_alpha.contiguous().data_ptr<float>(),
			dL_dout_color.contiguous().data_ptr<float>(),
			dL_dout_alpha.contiguous().data_ptr<float>(),
			dL_dmeans2D.contiguous().data_ptr<float>(),
			dL_dconic.contiguous().data_ptr<float>(),  
			dL_dopacity.contiguous().data_ptr<float>(),
			dL_dcolors.contiguous().data_ptr<float>(),
			dL_dmeans3D.contiguous().data_ptr<float>(),
			dL_dcov3D.contiguous().data_ptr<float>(),
			dL_dsh.contiguous().data_ptr<float>(),
			dL_dscales.contiguous().data_ptr<float>(),
			dL_drotations.contiguous().data_ptr<float>(),
			debug
		);
	}
	return;
}

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
) {
	cudaStream_t raw_stream = reinterpret_cast<cudaStream_t>(stream);
	const int P = means3D.size(1); // caution: with batch dim
	const int H = dL_dout_color.size(2);
	const int W = dL_dout_color.size(3);
	const int M = sh.size(0) == 0 ? 0 : sh.size(2);

	const float* background_ptr = background.data_ptr<float>() + batch_index * 3;
	const float* out_alpha_ptr = out_alpha.data_ptr<float>() + batch_index * 1 * H * W;
	const float* means3D_ptr = means3D.data_ptr<float>() + batch_index * P * 3;
	const float* sh_ptr = sh.data_ptr<float>() + batch_index * P * 3;
	const float* colors_ptr = nullptr;
	const float* scales_ptr = scales.data_ptr<float>() + batch_index * P * 3;
	const float* rotations_ptr = rotations.data_ptr<float>() + batch_index * P * 4;
	const float* cov3D_precomp_ptr = nullptr;
	const int* radii_ptr = radii.data_ptr<int>() + batch_index * P * 1;
	const float* dL_dout_color_ptr = dL_dout_color.data_ptr<float>() + batch_index * 3 * H * W;
	const float* dL_dout_alpha_ptr = dL_dout_alpha.data_ptr<float>() + batch_index * 1 * H * W;

	float* dL_dmeans2D_ptr = dL_dmeans2D.data_ptr<float>() + batch_index * P * 2;
	float* dL_dconic_ptr = dL_dconic.data_ptr<float>() + batch_index * P * 2 * 2;
	float* dL_dopacity_ptr = dL_dopacity.data_ptr<float>() + batch_index * P * 1;
	float* dL_dcolors_ptr = dL_dcolors.data_ptr<float>() + batch_index * P * 3;
	float* dL_dmeans3D_ptr = dL_dmeans3D.data_ptr<float>() + batch_index * P * 3;
	float* dL_dcov3D_ptr = dL_dcov3D.data_ptr<float>() + batch_index * P * 6;
	float* dL_dsh_ptr = dL_dsh.data_ptr<float>() + batch_index * P * 3; // TODO: fix hard code
	float* dL_dscales_ptr = dL_dscales.data_ptr<float>() + batch_index * P * 3;
	float* dL_drotations_ptr = dL_drotations.data_ptr<float>() + batch_index * P * 4;

	// cudaMemsetAsync(dL_dmeans2D_ptr, 0, P * 3 * sizeof(float), raw_stream);
	// cudaMemsetAsync(dL_dconic_ptr, 0, P * 2 * 2 * sizeof(float), raw_stream);
	// cudaMemsetAsync(dL_dopacity_ptr, 0, P * 1 * sizeof(float), raw_stream);
	// cudaMemsetAsync(dL_dcolors_ptr, 0, P * 3 * sizeof(float), raw_stream);
	// cudaMemsetAsync(dL_dmeans3D_ptr, 0, P * 3 * sizeof(float), raw_stream);
	// cudaMemsetAsync(dL_dcov3D_ptr, 0, P * 6 * sizeof(float), raw_stream);
	// cudaMemsetAsync(dL_dsh_ptr, 0, P * 3 * sizeof(float), raw_stream);
	// cudaMemsetAsync(dL_dscales_ptr, 0, P * 3 * sizeof(float), raw_stream);
	// cudaMemsetAsync(dL_drotations_ptr, 0, P * 4 * sizeof(float), raw_stream);
	
	if(P != 0) {
		CudaRasterizer::Rasterizer::backward(
			reinterpret_cast<cudaStream_t>(stream),
			P, degree, M, R,
			background_ptr,
			W, H, 
			means3D_ptr,
			sh_ptr,
			colors_ptr,
			scales_ptr,
			scale_modifier,
			rotations_ptr,
			cov3D_precomp_ptr,
			viewmatrix.contiguous().data_ptr<float>(),
			projmatrix.contiguous().data_ptr<float>(),
			campos.contiguous().data_ptr<float>(),
			tan_fovx,
			tan_fovy,
			radii_ptr,
			reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
			reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
			reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
			out_alpha_ptr,
			dL_dout_color_ptr,
			dL_dout_alpha_ptr,
			dL_dmeans2D_ptr,
			dL_dconic_ptr,  
			dL_dopacity_ptr,
			dL_dcolors_ptr,
			dL_dmeans3D_ptr,
			dL_dcov3D_ptr,
			dL_dsh_ptr,
			dL_dscales_ptr,
			dL_drotations_ptr,
			debug
		);
	}
	return;
}

torch::Tensor markVisible(
	torch::Tensor& means3D,
	torch::Tensor& viewmatrix,
	torch::Tensor& projmatrix
) { 
  const int P = means3D.size(0);
  
  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(torch::kBool));
 
  if(P != 0)
  {
	CudaRasterizer::Rasterizer::markVisible(P,
		means3D.contiguous().data_ptr<float>(),
		viewmatrix.contiguous().data_ptr<float>(),
		projmatrix.contiguous().data_ptr<float>(),
		present.contiguous().data_ptr<bool>());
  }
  
  return present;
}

torch::Tensor ComputeFaceTBNCUDA(const torch::Tensor& face_verts, const torch::Tensor& face_uvs) {
	if (face_verts.ndimension() != 4) { AT_ERROR("face_verts must have dimensions (B, F, 3, 3)"); }
	if (face_uvs.ndimension() != 3) { AT_ERROR("face_uvs must have dimensions (F, 3, 2)"); }
	const int B = face_verts.size(0);
	const int F = face_verts.size(1);
	torch::Tensor TBN = torch::zeros({B, F, 3, 3}, face_verts.options());
	if(F != 0 && B != 0) {
		CudaUtils::ComputeFaceTBN(
			B, F,
			face_verts.contiguous().data_ptr<float>(),
			face_uvs.contiguous().data_ptr<float>(),
			TBN.contiguous().data_ptr<float>()
		);
	}
	return TBN;
}

torch::Tensor MatrixToQuaternionCUDA(const torch::Tensor& matrix) {
	if (matrix.ndimension() != 3) { AT_ERROR("matrix must have dimensions (N, 3, 3)"); }
	const int P = matrix.size(0);
	torch::Tensor quaternion = torch::zeros({P, 4}, matrix.options());
	if(P != 0) {
		CudaUtils::MatrixToQuaternion::forward(
			P,
			matrix.contiguous().data_ptr<float>(),
			quaternion.contiguous().data_ptr<float>()
		);
	}
	return quaternion;
}

torch::Tensor MatrixToQuaternionBackwardCUDA(
	const torch::Tensor& grad_quaternion, 
	const torch::Tensor& matrix
) {
	const int P = matrix.size(0);
	torch::Tensor grad_matrix = torch::zeros({P, 3, 3}, matrix.options());
	if(P != 0) {
		CudaUtils::MatrixToQuaternion::backward(
			P,
			grad_quaternion.contiguous().data_ptr<float>(),
			matrix.contiguous().data_ptr<float>(),
			grad_matrix.contiguous().data_ptr<float>()
		);
	}
	return grad_matrix;
}

torch::Tensor QuaternionMultiplyCUDA(const torch::Tensor& q1, const torch::Tensor& q2) {
	if (q1.ndimension() != 2) { AT_ERROR("q1 must have dimensions (N, 4)"); }
	if (q2.ndimension() != 2) { AT_ERROR("q2 must have dimensions (N, 4)"); }
	const int P = q1.size(0);
	torch::Tensor product = torch::zeros({P, 4}, q1.options());
	if(P != 0) {
		CudaUtils::QuaternionMultiply::forward(
			P,
			q1.contiguous().data_ptr<float>(),
			q2.contiguous().data_ptr<float>(),
			product.contiguous().data_ptr<float>()
		);
	}
	return product;
}

std::tuple<torch::Tensor, torch::Tensor> QuaternionMultiplyBackwardCUDA(
	const torch::Tensor& grad_product, 
	const torch::Tensor& q1,
	const torch::Tensor& q2
) {
	const int P = q1.size(0);
	torch::Tensor grad_q1 = torch::zeros({P, 4}, q1.options());
	torch::Tensor grad_q2 = torch::zeros({P, 4}, q2.options());
	if(P != 0) {
		CudaUtils::QuaternionMultiply::backward(
			P,
			grad_product.contiguous().data_ptr<float>(),
			q1.contiguous().data_ptr<float>(),
			q2.contiguous().data_ptr<float>(),
			grad_q1.contiguous().data_ptr<float>(),
			grad_q2.contiguous().data_ptr<float>()
		);
	}
	return std::make_tuple(grad_q1, grad_q2);
}

void FastForwardCUDA(
	const float weight_threshold,
	const torch::Tensor& est_color,
	const torch::Tensor& est_weight,
	torch::Tensor& initialized,
	torch::Tensor& feature_dc
) {
	const int P = est_color.size(0);
	if (P != 0) {
		CudaRasterizer::Rasterizer::fastForward(
			P,
			weight_threshold,
			est_color.contiguous().data_ptr<float>(),
			est_weight.contiguous().data_ptr<float>(),
			initialized.contiguous().data_ptr<bool>(),
			feature_dc.contiguous().data_ptr<float>()
		);
	}
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> MeshBindingCUDA(
	const torch::Tensor& face_verts,
	const torch::Tensor& face_tbns,
	const torch::Tensor& binding_face_barys,
	const torch::Tensor& binding_face_ids,
	const torch::Tensor& gs_xyzs,
	const torch::Tensor& gs_rots
) {
	const int B = face_verts.size(0);
	const int N = gs_xyzs.size(1);
	const int F = face_verts.size(1);
	torch::Tensor transformed_gs_xyzs = torch::zeros({B, N, 3}, gs_xyzs.options());
	torch::Tensor transformed_gs_rots = torch::zeros({B, N, 4}, gs_rots.options());
	torch::Tensor binding_rotations = torch::zeros({B, N, 3, 3}, gs_rots.options());
	if(F != 0 && B != 0 && N != 0) {
		CudaUtils::MeshBinding::forward(
			B, N, F,
			face_verts.contiguous().data_ptr<float>(),
			face_tbns.contiguous().data_ptr<float>(),
			binding_face_barys.contiguous().data_ptr<float>(),
			binding_face_ids.contiguous().data_ptr<int>(),
			gs_xyzs.contiguous().data_ptr<float>(),
			gs_rots.contiguous().data_ptr<float>(),
			transformed_gs_xyzs.contiguous().data_ptr<float>(),
			transformed_gs_rots.contiguous().data_ptr<float>(),
			binding_rotations.contiguous().data_ptr<float>()
		);
	}
	return std::make_tuple(transformed_gs_xyzs, transformed_gs_rots, binding_rotations);
}

std::tuple<torch::Tensor, torch::Tensor> MeshBindingBackwardCUDA(
	const torch::Tensor& grad_transformed_gs_xyzs,
    const torch::Tensor& grad_transformed_gs_rots,
    const torch::Tensor& binding_rotations
) {
	const int B = grad_transformed_gs_xyzs.size(0);
	const int N = grad_transformed_gs_xyzs.size(1);
	torch::Tensor grad_gs_xyzs = torch::zeros({B, N, 3}, grad_transformed_gs_xyzs.options());
	torch::Tensor grad_gs_rots = torch::zeros({B, N, 4}, grad_transformed_gs_rots.options());
	if(B != 0 && N != 0) {
		CudaUtils::MeshBinding::backward(
			B, N,
			grad_transformed_gs_xyzs.contiguous().data_ptr<float>(),
			grad_transformed_gs_rots.contiguous().data_ptr<float>(),
			binding_rotations.contiguous().data_ptr<float>(),
			grad_gs_xyzs.contiguous().data_ptr<float>(),
			grad_gs_rots.contiguous().data_ptr<float>()
		);
	}
	return std::make_tuple(grad_gs_xyzs, grad_gs_rots);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> LinearBlendingCUDA(
	const torch::Tensor& blending_weights,
	const torch::Tensor& base_xyzs,
	const torch::Tensor& base_rots,
	const torch::Tensor& base_rgbs,
	const torch::Tensor& delta_xyzs,
	const torch::Tensor& delta_rots,
	const torch::Tensor& delta_rgbs
) {
	const int B = blending_weights.size(0);
	const int L = blending_weights.size(1);
	const int N = base_xyzs.size(0);

	torch::Tensor blended_xyzs = torch::zeros({B, N, 3}, base_xyzs.options());
	torch::Tensor blended_rots = torch::zeros({B, N, 4}, base_rots.options());
	torch::Tensor blended_rgbs = torch::zeros({B, N, 1, 3}, base_rgbs.options());

	if(B != 0 && N != 0 && L != 0) {
		CudaUtils::LinearBlending::forward(
			B, N, L,
			blending_weights.contiguous().data_ptr<float>(),
			base_xyzs.contiguous().data_ptr<float>(),
			base_rots.contiguous().data_ptr<float>(),
			base_rgbs.contiguous().data_ptr<float>(),
			delta_xyzs.contiguous().data_ptr<float>(),
			delta_rots.contiguous().data_ptr<float>(),
			delta_rgbs.contiguous().data_ptr<float>(),
			blended_xyzs.contiguous().data_ptr<float>(),
			blended_rots.contiguous().data_ptr<float>(),
			blended_rgbs.contiguous().data_ptr<float>()
		);
	}
	return std::make_tuple(blended_xyzs, blended_rots, blended_rgbs);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> LinearBlendingBackwardCUDA(
	const torch::Tensor& blending_weights,
	const torch::Tensor& delta_xyzs,
	const torch::Tensor& delta_rots,
	const torch::Tensor& delta_rgbs,
	const torch::Tensor& grad_blended_xyzs,
	const torch::Tensor& grad_blended_rots,
	const torch::Tensor& grad_blended_rgbs
) {
	const int B = blending_weights.size(0);
	const int L = blending_weights.size(1);
	const int N = grad_blended_xyzs.size(1);

	// torch::Tensor grad_blending_weights = torch::zeros({B, L, N}, blending_weights.options());
	torch::Tensor grad_blending_weights = torch::zeros({B, L}, blending_weights.options());
	torch::Tensor grad_delta_xyzs = torch::zeros({L, N, 3}, grad_blended_xyzs.options());
	torch::Tensor grad_delta_rots = torch::zeros({L, N, 4}, grad_blended_rots.options());
	torch::Tensor grad_delta_rgbs = torch::zeros({L, N, 1, 3}, grad_blended_rgbs.options());

	if(B != 0 && N != 0 && L != 0) {
		CudaUtils::LinearBlending::backward(
			B, N, L,
			blending_weights.contiguous().data_ptr<float>(),
			grad_blended_xyzs.contiguous().data_ptr<float>(),
			grad_blended_rots.contiguous().data_ptr<float>(),
			grad_blended_rgbs.contiguous().data_ptr<float>(),
			delta_xyzs.contiguous().data_ptr<float>(),
			delta_rots.contiguous().data_ptr<float>(),
			delta_rgbs.contiguous().data_ptr<float>(),
			grad_blending_weights.contiguous().data_ptr<float>(),
			grad_delta_xyzs.contiguous().data_ptr<float>(),
			grad_delta_rots.contiguous().data_ptr<float>(),
			grad_delta_rgbs.contiguous().data_ptr<float>()
		);
	}

	// grad_blending_weights = torch::sum(grad_blending_weights, 2);
	return std::make_tuple(grad_blending_weights, 
		grad_blended_xyzs, grad_blended_rots, grad_blended_rgbs, // grad_base_xxx is equivalent to grad_blended_xxx
		grad_delta_xyzs, grad_delta_rots, grad_delta_rgbs
	);
}