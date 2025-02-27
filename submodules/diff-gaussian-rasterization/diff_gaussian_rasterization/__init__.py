#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import Optional
from torch.nn import functional as F
import torch.nn as nn
import torch
from . import _C
from dataclasses import dataclass


def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)


def compute_face_tbn(face_vertices: torch.Tensor, face_uvs: torch.Tensor) -> torch.Tensor:
    return _C.compute_face_tbn(face_vertices, face_uvs)


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    batch_shape = matrix.shape[:-2]
    matrix = matrix.flatten(end_dim=-3) if len(batch_shape) > 0 else matrix.unsqueeze(0)
    raw_quaternion = _MatrixToQuaternion.apply(matrix)
    quaternion = F.normalize(raw_quaternion, dim=1)
    quaternion = quaternion.reshape(batch_shape + quaternion.shape[1:]) if len(batch_shape) > 0 else quaternion.squeeze(0)
    return quaternion


def quaternion_multiply(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    p_batch_shape = p.shape[:-1]
    q_batch_shape = q.shape[:-1]
    batch_shape = [max(a, b) for a, b in zip(p_batch_shape, q_batch_shape)]
    batch_shape.append(4)
    q = q.expand(batch_shape).flatten(end_dim=-2)
    p = p.expand(batch_shape).flatten(end_dim=-2)
    product = _QuaternionMultiply.apply(p, q)
    return product.reshape(batch_shape)


def fast_forward(
    weight_threshold: float,
	est_color: torch.Tensor,
	est_weight: torch.Tensor,
	initialized: torch.Tensor,
	feature_dc: torch.Tensor
):
    _C.fast_forward(weight_threshold, est_color, est_weight, initialized, feature_dc)


def mesh_binding(
    gs_xyzs: torch.Tensor, gs_rots: torch.Tensor, 
    face_verts: torch.Tensor, face_tbns: torch.Tensor, 
    binding_face_barys: torch.Tensor, binding_face_ids: torch.Tensor
):
    return _MeshBinding.apply(gs_xyzs, gs_rots, face_verts, face_tbns, binding_face_barys, binding_face_ids)


def linear_blending(
    blending_weights: torch.Tensor, 
    base_xyzs: torch.Tensor, base_rots: torch.Tensor, base_rgbs: torch.Tensor, 
    delta_xyzs: torch.Tensor, delta_rots: torch.Tensor, delta_rgbs: torch.Tensor
):
    return _LinearBlending.apply(
        blending_weights, 
        base_xyzs, base_rots, base_rgbs, 
        delta_xyzs, delta_rots, delta_rgbs
    )


class _LinearBlending(torch.autograd.Function):
    @staticmethod
    def forward(ctx, 
        blending_weights, 
        base_xyzs, base_rots, base_rgbs, 
        delta_xyzs, delta_rots, delta_rgbs
    ):
        blended_xyzs, blended_rots, blended_rgbs = _C.linear_blending(
            blending_weights, 
            base_xyzs, base_rots, base_rgbs, 
            delta_xyzs, delta_rots, delta_rgbs
        )
        ctx.save_for_backward(blending_weights, delta_xyzs, delta_rots, delta_rgbs)
        return blended_xyzs, blended_rots, blended_rgbs

    @staticmethod
    def backward(ctx, grad_blended_xyzs, grad_blended_rots, grad_blended_rgbs):
        blending_weights, delta_xyzs, delta_rots, delta_rgbs = ctx.saved_tensors
        grad_blending_weights, grad_base_xyzs, grad_base_rots, grad_base_rgbs, grad_delta_xyzs, grad_delta_rots, grad_delta_rgbs = \
            _C.linear_blending_backward(
                blending_weights, 
                delta_xyzs, delta_rots, delta_rgbs,
                grad_blended_xyzs, grad_blended_rots, grad_blended_rgbs
            )
        return grad_blending_weights, grad_base_xyzs, grad_base_rots, grad_base_rgbs, grad_delta_xyzs, grad_delta_rots, grad_delta_rgbs


class _MeshBinding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gs_xyzs, gs_rots, face_verts, face_tbns, binding_face_barys, binding_face_ids):
        transformed_gs_xyzs, transformed_gs_rots, binding_rotations = _C.mesh_binding(
            face_verts, face_tbns, binding_face_barys, binding_face_ids, gs_xyzs, gs_rots)
        ctx.save_for_backward(binding_rotations)
        return transformed_gs_xyzs, transformed_gs_rots

    @staticmethod
    def backward(ctx, grad_transformed_gs_xyzs, grad_transformed_gs_rots):
        binding_rotations, = ctx.saved_tensors
        grad_gs_xyzs, grad_gs_rots = _C.mesh_binding_backward(grad_transformed_gs_xyzs, grad_transformed_gs_rots, binding_rotations)
        return grad_gs_xyzs, grad_gs_rots, None, None, None, None


class _QuaternionMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p, q):
        product = _C.quaternion_multiply(p, q)
        ctx.save_for_backward(p, q)
        return product

    @staticmethod
    def backward(ctx, grad_product):
        p, q = ctx.saved_tensors
        grad_p, grad_q = _C.quaternion_multiply_backward(grad_product, p, q)
        return grad_p, grad_q


class _MatrixToQuaternion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, matrix):
        quaternion = _C.matrix_to_quaternion(matrix)
        ctx.save_for_backward(matrix)
        return quaternion

    @staticmethod
    def backward(ctx, grad_quaternion):
        matrix, = ctx.saved_tensors
        grad_matrix = _C.matrix_to_quaternion_backward(grad_quaternion, matrix)
        return grad_matrix


@dataclass
class GaussianRasterizationSettings():
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool


class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D: torch.Tensor,
        means2D: torch.Tensor,
        sh: torch.Tensor,
        colors_precomp: torch.Tensor,
        opacities: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        cov3Ds_precomp: torch.Tensor,
        target_image: torch.Tensor,
        raster_settings: GaussianRasterizationSettings,
    ):
        num_gs = means3D.shape[0]
        num_ch = 3 # TODO: hard code, keep same with NUM_CHANNELS in C++
        color = torch.zeros([num_ch, raster_settings.image_height, raster_settings.image_width], device='cuda', dtype=torch.float32)
        alpha = torch.zeros([1, raster_settings.image_height, raster_settings.image_width], device='cuda', dtype=torch.float32)
        est_color = torch.zeros([num_gs, num_ch], device='cuda', dtype=torch.float32)
        est_weight = torch.zeros([num_gs], device='cuda', dtype=torch.float32)
        radii = torch.zeros([num_gs], device='cuda', dtype=torch.int32)

        geom_buffer = torch.empty(0, dtype=torch.uint8, device='cuda') 
        binning_buffer = torch.empty(0, dtype=torch.uint8, device='cuda') 
        img_buffer = torch.empty(0, dtype=torch.uint8, device='cuda') 
        
        stream = torch.cuda.current_stream().cuda_stream
        num_rendered = _C.rasterize_gaussians(
            stream,
            raster_settings.bg, 
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            target_image,
            geom_buffer,
            binning_buffer,
            img_buffer,
            color,
            alpha,
            est_color,
            est_weight,
            radii,
            raster_settings.prefiltered,
            raster_settings.debug
        )

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geom_buffer, binning_buffer, img_buffer, alpha)
        return color, alpha, est_color, est_weight, radii

    @staticmethod
    def backward(ctx, grad_out_color, grad_out_alpha, grad_est_color, grad_est_weight, grad_out_radii):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer, alpha = ctx.saved_tensors

        num_gs = means3D.shape[0]
        num_ch = 3 # TODO: hard code, keep same with NUM_CHANNELS in C++
        num_sh = 0 if sh.size(0) == 0 else sh.size(1)
        grad_means3D = torch.zeros([num_gs, 3], device='cuda', dtype=torch.float32)
        grad_means2D = torch.zeros([num_gs, 3], device='cuda', dtype=torch.float32)
        grad_colors = torch.zeros([num_gs, num_ch], device='cuda', dtype=torch.float32)
        grad_conic = torch.zeros([num_gs, 2, 2], device='cuda', dtype=torch.float32)
        grad_opacities = torch.zeros([num_gs, 1], device='cuda', dtype=torch.float32)
        grad_cov3d = torch.zeros([num_gs, 6], device='cuda', dtype=torch.float32)
        grad_sh = torch.zeros([num_gs, num_sh, 3], device='cuda', dtype=torch.float32)
        grad_scales = torch.zeros([num_gs, 3], device='cuda', dtype=torch.float32)
        grad_rotations = torch.zeros([num_gs, 4], device='cuda', dtype=torch.float32)

        current_stream = torch.cuda.current_stream()

        # Restructure args as C++ method expects them
        _C.rasterize_gaussians_backward(
            current_stream.cuda_stream,
            raster_settings.bg,
            means3D, 
            radii, 
            colors_precomp, 
            scales, 
            rotations, 
            raster_settings.scale_modifier, 
            cov3Ds_precomp, 
            raster_settings.viewmatrix, 
            raster_settings.projmatrix, 
            raster_settings.tanfovx, 
            raster_settings.tanfovy, 
            grad_out_color, 
            grad_out_alpha,
            sh, 
            raster_settings.sh_degree, 
            raster_settings.campos,
            geomBuffer,
            num_rendered,
            binningBuffer,
            imgBuffer,
            alpha,
            grad_means3D,
            grad_means2D,
            grad_colors,
            grad_conic,
            grad_opacities,
            grad_cov3d,
            grad_sh,
            grad_scales,
            grad_rotations,
            raster_settings.debug
        )

        return (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3d,
            None,
            None,
        )
    

@dataclass
class BatchRasterizationTensors():
    color: torch.Tensor
    alpha: torch.Tensor
    est_color: torch.Tensor
    est_weight: torch.Tensor
    radii: torch.Tensor
    grad_means3D: torch.Tensor
    grad_means2D: torch.Tensor
    grad_colors: torch.Tensor
    grad_conic: torch.Tensor
    grad_opacities: torch.Tensor
    grad_cov3d: torch.Tensor
    grad_sh: torch.Tensor
    grad_scales: torch.Tensor
    grad_rotations: torch.Tensor
    colors_precomp: torch.Tensor
    cov3Ds_precomp: torch.Tensor
    buffer_list: list[torch.Tensor]
    stream_list: list[torch.cuda.Stream]


class _BatchRasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward_naive(
        ctx,
        means3D: torch.Tensor,
        means2D: torch.Tensor,
        sh: torch.Tensor,
        opacities: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        target_image: torch.Tensor,
        raster_settings: GaussianRasterizationSettings,
        raster_tensors: BatchRasterizationTensors
    ):
        batch_size = means3D.size(0)

        raster_tensors.color.zero_()
        raster_tensors.alpha.zero_()
        raster_tensors.est_color.zero_()
        raster_tensors.est_weight.zero_()
        raster_tensors.radii.zero_()

        current_stream = torch.cuda.current_stream()

        num_rendered = [0 for _ in range(batch_size)]
        for batch_index in range(batch_size):
            geom_buffer = raster_tensors.buffer_list[3 * batch_index]
            binning_buffer = raster_tensors.buffer_list[3 * batch_index + 1]
            img_buffer = raster_tensors.buffer_list[3 * batch_index + 2]
            num_rendered[batch_index] = _C.rasterize_gaussians(
                current_stream.cuda_stream,
                raster_settings.bg[batch_index], 
                means3D[batch_index],
                raster_tensors.colors_precomp,
                opacities[batch_index],
                scales[batch_index],
                rotations[batch_index],
                raster_settings.scale_modifier,
                raster_tensors.cov3Ds_precomp,
                raster_settings.viewmatrix,
                raster_settings.projmatrix,
                raster_settings.tanfovx,
                raster_settings.tanfovy,
                raster_settings.image_height,
                raster_settings.image_width,
                sh[batch_index],
                raster_settings.sh_degree,
                raster_settings.campos,
                target_image,
                geom_buffer,
                binning_buffer,
                img_buffer,
                raster_tensors.color[batch_index],
                raster_tensors.alpha[batch_index],
                raster_tensors.est_color[batch_index],
                raster_tensors.est_weight[batch_index],
                raster_tensors.radii[batch_index],
                raster_settings.prefiltered,
                raster_settings.debug
            )

        # Keep relevant tensors for backward
        ctx.raster_tensors = raster_tensors
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(means3D, scales, rotations, sh)
        return (
            raster_tensors.color[:batch_size], 
            raster_tensors.alpha[:batch_size], 
            raster_tensors.est_color[:batch_size], 
            raster_tensors.est_weight[:batch_size], 
            raster_tensors.radii[:batch_size]
        )
    
    @staticmethod
    def forward_wi_sync(
        ctx,
        means3D: torch.Tensor,
        means2D: torch.Tensor,
        sh: torch.Tensor,
        opacities: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        target_image: torch.Tensor,
        raster_settings: GaussianRasterizationSettings,
        raster_tensors: BatchRasterizationTensors
    ):
        batch_size = means3D.size(0)
        num_rendered = torch.zeros([batch_size], dtype=torch.int32, device=means3D.device)

        means3D = means3D.contiguous()
        sh = sh.contiguous()
        opacities = opacities.contiguous()
        scales = scales.contiguous()
        rotations = rotations.contiguous()
        target_image = target_image.contiguous()

        current_stream = torch.cuda.current_stream()

        for batch_index in range(batch_size):
            geom_buffer = raster_tensors.buffer_list[batch_index * 3]
            _C.batch_rasterize_gaussians1(
                current_stream.cuda_stream,
                batch_index,
                num_rendered,
                means3D,
                raster_tensors.colors_precomp, # empty
                opacities,
                scales,
                rotations,
                raster_settings.scale_modifier,
                raster_tensors.cov3Ds_precomp, # empty
                raster_settings.viewmatrix,
                raster_settings.projmatrix,
                raster_settings.tanfovx,
                raster_settings.tanfovy,
                raster_settings.image_height,
                raster_settings.image_width,
                sh,
                raster_settings.sh_degree,
                raster_settings.campos,
                geom_buffer, 
                raster_tensors.radii,
                raster_settings.prefiltered,
                raster_settings.debug
            )

        num_rendered = num_rendered.to(device='cpu', non_blocking=False)

        P = means3D.size(1)
        for batch_index in range(batch_size):
            geom_buffer = raster_tensors.buffer_list[batch_index * 3]
            binning_buffer = raster_tensors.buffer_list[batch_index * 3 + 1]
            img_buffer = raster_tensors.buffer_list[batch_index * 3 + 2]
            _C.batch_rasterize_gaussians2(
                current_stream.cuda_stream,
                batch_index,
                P, num_rendered,
                raster_settings.bg, # caution: background has batch dim
                raster_tensors.colors_precomp, # empty
                raster_settings.image_height,
                raster_settings.image_width,
                target_image,
                geom_buffer, 
                binning_buffer, 
                img_buffer,
                raster_tensors.color,
                raster_tensors.alpha,
                raster_tensors.est_color,
                raster_tensors.est_weight,
                raster_tensors.radii,
                raster_settings.debug
            )

        # Keep relevant tensors for backward
        ctx.raster_tensors = raster_tensors
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered.tolist()
        ctx.save_for_backward(means3D, scales, rotations, sh)
        return (
            raster_tensors.color[:batch_size], 
            raster_tensors.alpha[:batch_size], 
            raster_tensors.est_color[:batch_size], 
            raster_tensors.est_weight[:batch_size], 
            raster_tensors.radii[:batch_size]
        )

    @staticmethod
    def forward_wi_stream(
        ctx,
        means3D: torch.Tensor,
        means2D: torch.Tensor,
        sh: torch.Tensor,
        opacities: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        target_image: torch.Tensor,
        raster_settings: GaussianRasterizationSettings,
        raster_tensors: BatchRasterizationTensors
    ):
        batch_size = means3D.size(0)

        raster_tensors.color.zero_()
        raster_tensors.alpha.zero_()
        raster_tensors.est_color.zero_()
        raster_tensors.est_weight.zero_()
        raster_tensors.radii.zero_()

        raster_tensors.grad_means3D.zero_()
        raster_tensors.grad_means2D.zero_()
        raster_tensors.grad_colors.zero_()
        raster_tensors.grad_conic.zero_()
        raster_tensors.grad_opacities.zero_()
        raster_tensors.grad_cov3d.zero_()
        raster_tensors.grad_sh.zero_()
        raster_tensors.grad_scales.zero_()
        raster_tensors.grad_rotations.zero_()

        current_stream = torch.cuda.current_stream()
        stream_size = len(raster_tensors.stream_list)

        for s in range(stream_size): # caution: must wait current stream
            raster_tensors.stream_list[s].wait_stream(current_stream)

        num_rendered = [0 for _ in range(batch_size)]
        for batch_index in range(batch_size):
            geom_buffer = raster_tensors.buffer_list[3 * batch_index]
            binning_buffer = raster_tensors.buffer_list[3 * batch_index + 1]
            img_buffer = raster_tensors.buffer_list[3 * batch_index + 2]
            stream = raster_tensors.stream_list[batch_index % stream_size]
            num_rendered[batch_index] = _C.rasterize_gaussians(
                stream.cuda_stream,
                raster_settings.bg[batch_index], 
                means3D[batch_index],
                raster_tensors.colors_precomp,
                opacities[batch_index],
                scales[batch_index],
                rotations[batch_index],
                raster_settings.scale_modifier,
                raster_tensors.cov3Ds_precomp,
                raster_settings.viewmatrix,
                raster_settings.projmatrix,
                raster_settings.tanfovx,
                raster_settings.tanfovy,
                raster_settings.image_height,
                raster_settings.image_width,
                sh[batch_index],
                raster_settings.sh_degree,
                raster_settings.campos,
                target_image,
                geom_buffer,
                binning_buffer,
                img_buffer,
                raster_tensors.color[batch_index],
                raster_tensors.alpha[batch_index],
                raster_tensors.est_color[batch_index],
                raster_tensors.est_weight[batch_index],
                raster_tensors.radii[batch_index],
                raster_settings.prefiltered,
                raster_settings.debug
            )

        for s in range(stream_size): # caution: current stream must wait batch streams
            current_stream.wait_stream(raster_tensors.stream_list[s])

        # Keep relevant tensors for backward
        ctx.raster_tensors = raster_tensors
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(means3D, scales, rotations, sh)
        return (
            raster_tensors.color[:batch_size], 
            raster_tensors.alpha[:batch_size], 
            raster_tensors.est_color[:batch_size], 
            raster_tensors.est_weight[:batch_size], 
            raster_tensors.radii[:batch_size]
        )

    @staticmethod
    def backward_naive(ctx, grad_out_color, grad_out_alpha, grad_est_color, grad_est_weight, grad_out_radii):
        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        raster_tensors = ctx.raster_tensors
        means3D, scales, rotations, sh = ctx.saved_tensors

        batch_size = means3D.shape[0]
        raster_tensors.grad_means3D.zero_()
        raster_tensors.grad_means2D.zero_()
        raster_tensors.grad_colors.zero_()
        raster_tensors.grad_conic.zero_()
        raster_tensors.grad_opacities.zero_()
        raster_tensors.grad_cov3d.zero_()
        raster_tensors.grad_sh.zero_()
        raster_tensors.grad_scales.zero_()
        raster_tensors.grad_rotations.zero_()

        current_stream = torch.cuda.current_stream()

        for batch_index in range(batch_size):
            geom_buffer = raster_tensors.buffer_list[3 * batch_index]
            binning_buffer = raster_tensors.buffer_list[3 * batch_index + 1]
            img_buffer = raster_tensors.buffer_list[3 * batch_index + 2]
            _C.rasterize_gaussians_backward(
                current_stream.cuda_stream,
                raster_settings.bg[batch_index], # caution: background has batch dim
                means3D[batch_index], 
                raster_tensors.radii[batch_index], 
                raster_tensors.colors_precomp, # empty
                scales[batch_index], 
                rotations[batch_index], 
                raster_settings.scale_modifier, 
                raster_tensors.cov3Ds_precomp, # empty
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color[batch_index], 
                grad_out_alpha[batch_index],
                sh[batch_index], 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geom_buffer,
                num_rendered[batch_index],
                binning_buffer,
                img_buffer,
                raster_tensors.alpha[batch_index],
                raster_tensors.grad_means3D[batch_index],
                raster_tensors.grad_means2D[batch_index],
                raster_tensors.grad_colors[batch_index],
                raster_tensors.grad_conic[batch_index],
                raster_tensors.grad_opacities[batch_index],
                raster_tensors.grad_cov3d[batch_index],
                raster_tensors.grad_sh[batch_index],
                raster_tensors.grad_scales[batch_index],
                raster_tensors.grad_rotations[batch_index],
                raster_settings.debug
            )

        return (
            raster_tensors.grad_means3D[:batch_size],
            raster_tensors.grad_means2D[:batch_size],
            raster_tensors.grad_sh[:batch_size],
            raster_tensors.grad_opacities[:batch_size],
            raster_tensors.grad_scales[:batch_size],
            raster_tensors.grad_rotations[:batch_size],
            None,
            None,
            None
        )

    @staticmethod
    def forward(
        ctx,
        means3D: torch.Tensor,
        means2D: torch.Tensor,
        sh: torch.Tensor,
        opacities: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        target_image: torch.Tensor,
        raster_settings: GaussianRasterizationSettings,
        raster_tensors: BatchRasterizationTensors
    ):
        batch_size = means3D.size(0)
        num_rendered = torch.zeros([batch_size], dtype=torch.int32, device=means3D.device)

        means3D = means3D.contiguous()
        sh = sh.contiguous()
        opacities = opacities.contiguous()
        scales = scales.contiguous()
        rotations = rotations.contiguous()
        target_image = target_image.contiguous()

        raster_tensors.color.zero_()
        raster_tensors.alpha.zero_()
        raster_tensors.est_color.zero_()
        raster_tensors.est_weight.zero_()
        raster_tensors.radii.zero_()

        raster_tensors.grad_means3D.zero_()
        raster_tensors.grad_means2D.zero_()
        raster_tensors.grad_colors.zero_()
        raster_tensors.grad_conic.zero_()
        raster_tensors.grad_opacities.zero_()
        raster_tensors.grad_cov3d.zero_()
        raster_tensors.grad_sh.zero_()
        raster_tensors.grad_scales.zero_()
        raster_tensors.grad_rotations.zero_()

        current_stream = torch.cuda.current_stream()
        stream_size = len(raster_tensors.stream_list)

        for s in range(stream_size): # caution: must wait current stream
            raster_tensors.stream_list[s].wait_stream(current_stream)

        for batch_index in range(batch_size):
            geom_buffer = raster_tensors.buffer_list[batch_index * 3]
            stream = raster_tensors.stream_list[batch_index % stream_size].cuda_stream
            _C.batch_rasterize_gaussians1(
                stream,
                batch_index,
                num_rendered,
                means3D,
                raster_tensors.colors_precomp, # empty
                opacities,
                scales,
                rotations,
                raster_settings.scale_modifier,
                raster_tensors.cov3Ds_precomp, # empty
                raster_settings.viewmatrix,
                raster_settings.projmatrix,
                raster_settings.tanfovx,
                raster_settings.tanfovy,
                raster_settings.image_height,
                raster_settings.image_width,
                sh,
                raster_settings.sh_degree,
                raster_settings.campos,
                geom_buffer, 
                raster_tensors.radii,
                raster_settings.prefiltered,
                raster_settings.debug
            )

        for s in range(stream_size):
            raster_tensors.stream_list[s].synchronize()
        num_rendered = num_rendered.to(device='cpu', non_blocking=False)

        P = means3D.size(1)
        for batch_index in range(batch_size):
            geom_buffer = raster_tensors.buffer_list[batch_index * 3]
            binning_buffer = raster_tensors.buffer_list[batch_index * 3 + 1]
            img_buffer = raster_tensors.buffer_list[batch_index * 3 + 2]
            stream = raster_tensors.stream_list[batch_index % stream_size]
            _C.batch_rasterize_gaussians2(
                stream.cuda_stream,
                batch_index,
                P, num_rendered,
                raster_settings.bg, # caution: background has batch dim
                raster_tensors.colors_precomp, # empty
                raster_settings.image_height,
                raster_settings.image_width,
                target_image,
                geom_buffer, 
                binning_buffer, 
                img_buffer,
                raster_tensors.color,
                raster_tensors.alpha,
                raster_tensors.est_color,
                raster_tensors.est_weight,
                raster_tensors.radii,
                raster_settings.debug
            )

        for s in range(stream_size): # caution: current stream must wait batch streams
            current_stream.wait_stream(raster_tensors.stream_list[s])

        # Keep relevant tensors for backward
        ctx.raster_tensors = raster_tensors
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered.tolist()
        ctx.save_for_backward(means3D, scales, rotations, sh)
        return (
            raster_tensors.color[:batch_size], 
            raster_tensors.alpha[:batch_size], 
            raster_tensors.est_color[:batch_size], 
            raster_tensors.est_weight[:batch_size], 
            raster_tensors.radii[:batch_size]
        )
    
    @staticmethod
    def backward(ctx, grad_out_color, grad_out_alpha, grad_est_color, grad_est_weight, grad_out_radii):
        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        raster_tensors = ctx.raster_tensors
        means3D, scales, rotations, sh = ctx.saved_tensors

        batch_size = means3D.shape[0]
        grad_out_color = grad_out_color.contiguous()
        grad_out_alpha = grad_out_alpha.contiguous()

        current_stream = torch.cuda.current_stream()
        stream_size = len(raster_tensors.stream_list)

        for s in range(stream_size): # caution: must wait current stream
            raster_tensors.stream_list[s].wait_stream(current_stream)

        for batch_index in range(batch_size):
            geom_buffer = raster_tensors.buffer_list[3 * batch_index]
            binning_buffer = raster_tensors.buffer_list[3 * batch_index + 1]
            img_buffer = raster_tensors.buffer_list[3 * batch_index + 2]
            stream = raster_tensors.stream_list[batch_index % stream_size].cuda_stream
            _C.batch_rasterize_gaussians_backward(
                stream,
                batch_index,
                raster_settings.bg, # caution: background has batch dim
                means3D, 
                raster_tensors.radii, 
                raster_tensors.colors_precomp, # empty
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                raster_tensors.cov3Ds_precomp, # empty
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color, 
                grad_out_alpha,
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geom_buffer,
                num_rendered[batch_index],
                binning_buffer,
                img_buffer,
                raster_tensors.alpha,
                raster_tensors.grad_means3D,
                raster_tensors.grad_means2D,
                raster_tensors.grad_colors,
                raster_tensors.grad_conic,
                raster_tensors.grad_opacities,
                raster_tensors.grad_cov3d,
                raster_tensors.grad_sh,
                raster_tensors.grad_scales,
                raster_tensors.grad_rotations,
                raster_settings.debug
            )

        for s in range(stream_size): # caution: current stream must wait batch streams
            current_stream.wait_stream(raster_tensors.stream_list[s])

        return (
            raster_tensors.grad_means3D[:batch_size],
            raster_tensors.grad_means2D[:batch_size],
            raster_tensors.grad_sh[:batch_size],
            raster_tensors.grad_opacities[:batch_size],
            raster_tensors.grad_scales[:batch_size],
            raster_tensors.grad_rotations[:batch_size],
            None,
            None,
            None
        )


class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings: GaussianRasterizationSettings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions: torch.Tensor):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, 
        means3D: torch.Tensor, 
        means2D: torch.Tensor, 
        opacities: torch.Tensor, 
        shs: Optional[torch.Tensor] = None, 
        colors_precomp: Optional[torch.Tensor] = None, 
        scales: Optional[torch.Tensor] = None, 
        rotations: Optional[torch.Tensor] = None, 
        target_image: Optional[torch.Tensor] = None, 
        cov3D_precomp: Optional[torch.Tensor] = None
    ):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None: shs = torch.Tensor([])
        if colors_precomp is None: colors_precomp = torch.Tensor([])
        if scales is None: scales = torch.Tensor([])
        if rotations is None: rotations = torch.Tensor([])
        if cov3D_precomp is None: cov3D_precomp = torch.Tensor([])
        if target_image is None: target_image = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return _RasterizeGaussians.apply(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            target_image,
            raster_settings,
        )
    

class BatchGaussianRasterizer(nn.Module):
    def __init__(self, 
        max_gaussian_size: int,
        max_batch_size: int,
        raster_settings: GaussianRasterizationSettings
    ):
        super().__init__()
        channel_size = 3 # TODO: hard code, keep same with NUM_CHANNELS in C++
        sh_size = (raster_settings.sh_degree + 1) ** 2
        self.max_gaussian_size = max_gaussian_size
        self.max_batch_size = max_batch_size
        self.raster_settings = raster_settings
        self.raster_tensors = BatchRasterizationTensors(
            color= torch.zeros([max_batch_size, channel_size, raster_settings.image_height, raster_settings.image_width], device='cuda', dtype=torch.float32),
            alpha = torch.zeros([max_batch_size, 1, raster_settings.image_height, raster_settings.image_width], device='cuda', dtype=torch.float32),
            est_color = torch.zeros([max_batch_size, max_gaussian_size, channel_size], device='cuda', dtype=torch.float32),
            est_weight = torch.zeros([max_batch_size, max_gaussian_size], device='cuda', dtype=torch.float32),
            radii = torch.zeros([max_batch_size, max_gaussian_size], device='cuda', dtype=torch.int32),
            grad_means3D = torch.zeros([max_batch_size, max_gaussian_size, 3], device='cuda', dtype=torch.float32),
            grad_means2D = torch.zeros([max_batch_size, max_gaussian_size, 3], device='cuda', dtype=torch.float32),
            grad_colors = torch.zeros([max_batch_size, max_gaussian_size, channel_size], device='cuda', dtype=torch.float32),
            grad_conic = torch.zeros([max_batch_size, max_gaussian_size, 2, 2], device='cuda', dtype=torch.float32),
            grad_opacities = torch.zeros([max_batch_size, max_gaussian_size, 1], device='cuda', dtype=torch.float32),
            grad_cov3d = torch.zeros([max_batch_size, max_gaussian_size, 6], device='cuda', dtype=torch.float32),
            grad_sh = torch.zeros([max_batch_size, max_gaussian_size, sh_size, 3], device='cuda', dtype=torch.float32),
            grad_scales = torch.zeros([max_batch_size, max_gaussian_size, 3], device='cuda', dtype=torch.float32),
            grad_rotations = torch.zeros([max_batch_size, max_gaussian_size, 4], device='cuda', dtype=torch.float32),
            colors_precomp = torch.Tensor([]),
            cov3Ds_precomp = torch.Tensor([]),
            buffer_list = [torch.empty(0, device='cuda', dtype=torch.uint8) for _ in range(max_batch_size * 3)],
            stream_list = [torch.cuda.Stream() for _ in range(3)]
        )

    def forward(self, 
        means3D: torch.Tensor, 
        means2D: torch.Tensor, 
        opacities: torch.Tensor, 
        shs: torch.Tensor, 
        scales: torch.Tensor, 
        rotations: torch.Tensor,
        target_image: Optional[torch.Tensor] = None
    ):
        assert means3D.ndim == 3
        assert means2D.ndim == 3
        assert opacities.ndim == 3
        assert shs.ndim == 4
        assert scales.ndim == 3
        assert rotations.ndim == 3
        if target_image is not None: assert target_image.ndim == 4
        else: target_image = torch.Tensor([])

        assert means3D.size(0) <= self.max_batch_size
        assert means3D.size(1) == self.max_gaussian_size

        # Invoke C++/CUDA rasterization routine
        return _BatchRasterizeGaussians.apply(
            means3D,
            means2D,
            shs,
            opacities,
            scales,
            rotations,
            target_image,
            self.raster_settings,
            self.raster_tensors
        )

