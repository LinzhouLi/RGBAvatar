from typing import Union, Tuple

import torch
import nvdiffrast.torch as dr

def render_texture(
    uvs: torch.Tensor, # [V1, 2]
    uv_faces: torch.Tensor, # [F, 3]
    attrs: torch.Tensor, # [V2, C]
    attr_faces: torch.Tensor, # [F, 3]
    size: Tuple[int, int], # (H, W)
    glctx: Union[dr.RasterizeGLContext, dr.RasterizeCudaContext]
) -> torch.Tensor:
    hack_verts_clip = uvs * 2.0 - 1.0 # (0, 1) => (-1, 1)
    hack_verts_clip = torch.nn.functional.pad(hack_verts_clip, [0, 1], value=0.0)
    hack_verts_clip = torch.nn.functional.pad(hack_verts_clip, [0, 1], value=1.0)

    uv_faces = uv_faces.to(torch.int32)
    rast_out, _ = dr.rasterize(glctx, hack_verts_clip.unsqueeze(0), uv_faces, resolution=(size[0], size[1]))

    attr_faces = attr_faces.to(torch.int32) # caution: attr_faces is different from uv_faces
    interp_out, _ = dr.interpolate(attrs, rast_out, attr_faces)
    return interp_out.squeeze(0) # [H, W, C]


def compute_rast_info(
    uvs: torch.Tensor, # [V1, 2]
    uv_faces: torch.Tensor, # [F, 3]
    size: Tuple[int, int], # (H, W)
    glctx: Union[dr.RasterizeGLContext, dr.RasterizeCudaContext]
) -> tuple[torch.Tensor, torch.Tensor]:
    hack_verts_clip = uvs * 2.0 - 1.0 # (0, 1) => (-1, 1)
    hack_verts_clip = torch.nn.functional.pad(hack_verts_clip, [0, 1], value=0.0)
    hack_verts_clip = torch.nn.functional.pad(hack_verts_clip, [0, 1], value=1.0)

    uv_faces = uv_faces.to(torch.int32)
    rast_out, _ = dr.rasterize(glctx, hack_verts_clip.unsqueeze(0), uv_faces, resolution=(size[0], size[1]))

    rast_out = rast_out.squeeze(0)
    face_uv = rast_out[..., :2]
    face_id = rast_out[..., 3:] # face_id == 0 when no triangle
    return face_uv, face_id.to(torch.int) # [H, W, 2], [H, W, 1]