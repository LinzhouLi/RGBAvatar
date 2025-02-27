from typing import Optional

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, BatchGaussianRasterizer
from camera import Camera
from . import GaussianAttributes


class BatchGaussianRenderer:
    def __init__(self,
        bg_color: torch.Tensor,
        static_camera: Camera,
        num_gaussians: int,
        batch_size: int,
        sh_degree: int = 0
    ) -> None:
        tanfovx = math.tan(static_camera.fov_x * 0.5)
        tanfovy = math.tan(static_camera.fov_y * 0.5)

        viewmatrix = static_camera.get_w2v
        projmatrix = static_camera.get_full_proj
        raster_settings = GaussianRasterizationSettings(
            image_height=int(static_camera.height),
            image_width=int(static_camera.width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=1.0,
            viewmatrix=viewmatrix.transpose(0, 1).contiguous().cuda(),
            projmatrix=projmatrix.transpose(0, 1).contiguous().cuda(),
            sh_degree=sh_degree,
            campos=static_camera.get_pos.contiguous().cuda(),
            prefiltered=False,
            debug=False
        )

        self.batch_size = batch_size
        self.rasterizer = BatchGaussianRasterizer(
            max_gaussian_size=num_gaussians,
            max_batch_size=batch_size,
            raster_settings=raster_settings
        )


    def render(self,
        bg_color: torch.Tensor,
        gs: GaussianAttributes,
        target_image: Optional[torch.Tensor] = None
    ):
        screenspace_points = torch.zeros_like(gs.xyz, dtype=gs.xyz.dtype, requires_grad=True, device=gs.xyz.device) + 0
        if screenspace_points.requires_grad: # requires_grad == False when inference
            screenspace_points.retain_grad()
        
        self.rasterizer.raster_settings.bg = bg_color
        gs.xyz = gs.xyz.contiguous()
        gs.sh = gs.sh.contiguous()
        gs.opacity = gs.opacity.contiguous()
        gs.scaling = gs.scaling.contiguous()
        gs.rotation = gs.rotation.contiguous()
        if target_image is not None:
            target_image = target_image.contiguous()

        color, alpha, est_color, est_weight, radii = self.rasterizer(
            means3D=gs.xyz,
            means2D=screenspace_points, 
            opacities=gs.opacity, 
            shs=gs.sh, 
            scales=gs.scaling, 
            rotations=gs.rotation,
            target_image=target_image
        )

        return {
            "color": color, 
            "alpha": alpha, 
            "est_color": est_color, 
            "est_weight": est_weight, 
            "radii": radii
        }