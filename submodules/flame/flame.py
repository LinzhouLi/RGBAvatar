# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2023 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: mica@tue.mpg.de

import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .lbs import lbs, blend_shapes, vertices2joints, lbs_transfer


mouth_v = torch.tensor([
    [0.034501, -0.07, 0.029227],
    [-0.034501, -0.07, 0.029227],
    [0.034501, -0.04, 0.029227],
    [-0.034501, -0.04, 0.029227]
], dtype=torch.float32)

mouth_uv = torch.tensor([
    [0.750000, 0.955000],
    [0.250000, 0.995000],
    [0.250000, 0.955000],
    [0.750000, 0.995000]
    # [0.750000, 0.955000],
    # [0.250000, 0.955000],
    # [0.750000, 0.995000],
    # [0.250000, 0.995000]
], dtype=torch.float32)

mouth_faces = torch.tensor([
    [3, 2, 1],
    [3, 4, 2]
], dtype=torch.int64) + 5022

mouth_uv_faces = torch.tensor([
    [1, 2, 3],
    [1, 4, 2]
    # [3, 2, 1],
    # [3, 4, 2]
], dtype=torch.int64) + 5117


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def to_tensor(array, dtype=torch.float32):
    if 'torch.tensor' not in str(type(array)):
        return torch.tensor(array, dtype=dtype)


def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


def rot_mat_to_euler(rot_mats):
    # Calculates rotation matrix to euler angles
    # Careful for extreme cases of eular angles like [0.0, pi, 0.0]

    sy = torch.sqrt(rot_mats[:, 0, 0] * rot_mats[:, 0, 0] +
                    rot_mats[:, 1, 0] * rot_mats[:, 1, 0])
    return torch.atan2(-rot_mats[:, 2, 0], sy)


class FLAME(nn.Module):
    """
    borrowed from https://github.com/soubhiksanyal/FLAME_PyTorch/blob/master/FLAME.py
    Given FLAME parameters for shape, pose, and expression, this class generates a differentiable FLAME function
    which outputs the a mesh and 2D/3D facial landmarks
    """

    def __init__(self, config):
        super(FLAME, self).__init__()
        print(f"[FLAME] Creating the 3DMM from {config.flame_geom_path}")
        with open(config.flame_geom_path, 'rb') as f:
            ss = pickle.load(f, encoding='latin1')
            self.flame_model = Struct(**ss)

        self.config = config
        dtype = config.dtype

        verts = torch.from_numpy(np.array(self.flame_model.v_template)).to(dtype)
        self.register_buffer('v_template', verts)
        self.register_buffer('mouth_v', mouth_v.to(dtype))

        faces = torch.from_numpy(np.array(self.flame_model.f)).to(torch.int64)
        self.register_buffer('faces', faces)

        # The shape components and expression
        shapedirs = torch.from_numpy(np.array(self.flame_model.shapedirs)).to(dtype)
        shapedirs = torch.cat([shapedirs[:, :, :config.num_shape_params], shapedirs[:, :, 300:300 + config.num_exp_params]], 2)
        self.register_buffer('shapedirs', shapedirs)
        # The pose components
        num_pose_basis = self.flame_model.posedirs.shape[-1]
        posedirs = np.reshape(self.flame_model.posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs', torch.from_numpy(np.array(posedirs)).to(dtype))
        #
        self.register_buffer('J_regressor', torch.from_numpy(np.array(self.flame_model.J_regressor.todense())).to(dtype))
        parents = torch.from_numpy(np.array(self.flame_model.kintree_table[0])).to(torch.long)
        parents[0] = -1
        self.register_buffer('parents', parents)
        self.register_buffer('lbs_weights', torch.from_numpy(np.array(self.flame_model.weights)).to(dtype))

        # Eyelid data
        self.register_buffer('l_eyelid', torch.from_numpy(np.load(config.l_eyelid_path)).to(dtype)[None])
        self.register_buffer('r_eyelid', torch.from_numpy(np.load(config.r_eyelid_path)).to(dtype)[None])

        # UV data
        flame_uv_data = np.load(config.flame_uv_path)
        uvs = torch.from_numpy(np.array(flame_uv_data['vt'])).to(dtype)
        self.register_buffer("uvs", uvs)
        uv_faces = torch.from_numpy(np.array(flame_uv_data['ft'])).to(torch.int64)
        self.register_buffer("uv_faces", uv_faces)

        # Landmark embeddings
        # lmk_embeddings = np.load(config.flame_lmk_path, allow_pickle=True, encoding='latin1')
        # lmk_embeddings = lmk_embeddings[()]
        # self.register_buffer('lmk_faces_idx', torch.from_numpy(np.array(lmk_embeddings['static_lmk_faces_idx'])).to(torch.int64))
        # self.register_buffer('lmk_bary_coords', torch.from_numpy(np.array(lmk_embeddings['static_lmk_bary_coords'])).to(dtype))
        # self.register_buffer('dynamic_lmk_faces_idx', torch.from_numpy(np.array(lmk_embeddings['dynamic_lmk_faces_idx'])).to(torch.int64))
        # self.register_buffer('dynamic_lmk_bary_coords', torch.from_numpy(np.array(lmk_embeddings['dynamic_lmk_bary_coords'])).to(dtype))

        if self.config.add_mouth: self.add_teeth()

        self.num_vertices = self.v_template.shape[0]
        self.num_faces = self.faces.shape[0]
        self.num_blend_basis = 129


    def add_teeth(self):
        device = self.v_template.device
        dtype = self.v_template.dtype

        vid_lip_outside_ring_upper = torch.tensor([
            1713, 1715, 1716, 1735, 1696, 1694, 1657, 3543, 2774, 2811, 2813, 2850, 2833, 2832, 2830
        ], device=device, dtype=torch.int64)
        vid_lip_outside_ring_lower = torch.tensor([
            1576, 1577, 1773, 1774, 1795, 1802, 1865, 3503, 2948, 2905, 2898, 2881, 2880, 2713, 2712
        ], device=device, dtype=torch.int64)

        v_lip_upper = self.v_template[vid_lip_outside_ring_upper]
        v_lip_lower = self.v_template[vid_lip_outside_ring_lower]

        # construct vertices for teeth
        mean_dist = (v_lip_upper - v_lip_lower).norm(dim=-1, keepdim=True).mean()
        v_teeth_middle = (v_lip_upper + v_lip_lower) / 2
        v_teeth_middle[:, 1] = v_teeth_middle[:, 1:2].mean(dim=0, keepdim=True)
        v_teeth_middle[:, 2] -= mean_dist * 2.0  # how far the teeth are from the lips

        # upper, front
        v_teeth_upper_edge = v_teeth_middle.clone() - torch.tensor([[0, mean_dist, 0]], device=device, dtype=dtype) * 1.0
        v_teeth_upper_root = v_teeth_upper_edge + torch.tensor([[0, mean_dist, 0]], device=device, dtype=dtype) * 1.0  # scale the height of teeth

        # lower, front
        v_teeth_lower_edge = v_teeth_middle.clone() + torch.tensor([[0, mean_dist, 0]], device=device, dtype=dtype) * 1.0
        v_teeth_lower_edge -= torch.tensor([[0, 0, mean_dist]], device=device, dtype=dtype) * 0.4  # slightly move the lower teeth to the back
        v_teeth_lower_root = v_teeth_lower_edge - torch.tensor([[0, mean_dist, 0]], device=device, dtype=dtype) * 1.0  # scale the height of teeth

        # concatenate to v_template
        num_verts_orig = self.v_template.shape[0]
        v_teeth = torch.cat([
            v_teeth_upper_root,  # num_verts_orig + 0-14 
            v_teeth_lower_root,  # num_verts_orig + 15-29
            v_teeth_upper_edge,  # num_verts_orig + 30-44
            v_teeth_lower_edge,  # num_verts_orig + 45-59
        ], dim=0)
        num_verts_teeth = v_teeth.shape[0]
        self.v_template = torch.cat([self.v_template, v_teeth], dim=0)

        vid_teeth_upper_root = torch.arange(0, 15, device=device, dtype=torch.int64) + num_verts_orig
        vid_teeth_lower_root = torch.arange(15, 30, device=device, dtype=torch.int64) + num_verts_orig
        vid_teeth_upper_edge = torch.arange(30, 45, device=device, dtype=torch.int64) + num_verts_orig
        vid_teeth_lower_edge = torch.arange(45, 60, device=device, dtype=torch.int64) + num_verts_orig
        
        vid_teeth_upper = torch.cat([vid_teeth_upper_root, vid_teeth_upper_edge], dim=0)
        vid_teeth_lower = torch.cat([vid_teeth_lower_root, vid_teeth_lower_edge], dim=0)
        vid_teeth = torch.cat([vid_teeth_upper, vid_teeth_lower], dim=0)

        # shapedirs copy from lips
        self.shapedirs = torch.cat([self.shapedirs, torch.zeros_like(self.shapedirs[:num_verts_teeth])], dim=0)
        shape_dirs_mean = (
            self.shapedirs[vid_lip_outside_ring_upper, :, :self.config.num_shape_params] + 
            self.shapedirs[vid_lip_outside_ring_lower, :, :self.config.num_shape_params]
        ) / 2
        self.shapedirs[vid_teeth_upper_root, :, :self.config.num_shape_params] = shape_dirs_mean
        self.shapedirs[vid_teeth_lower_root, :, :self.config.num_shape_params] = shape_dirs_mean
        self.shapedirs[vid_teeth_upper_edge, :, :self.config.num_shape_params] = shape_dirs_mean
        self.shapedirs[vid_teeth_lower_edge, :, :self.config.num_shape_params] = shape_dirs_mean

        # eyelid set to zero
        self.r_eyelid = torch.cat([self.r_eyelid, torch.zeros_like(self.r_eyelid[:, :num_verts_teeth])], dim=1)
        self.l_eyelid = torch.cat([self.l_eyelid, torch.zeros_like(self.l_eyelid[:, :num_verts_teeth])], dim=1)

        # posedirs set to zero
        posedirs = self.posedirs.reshape(len(self.parents) - 1, 9, num_verts_orig, 3)  # (J*9, V*3) -> (J, 9, V, 3)
        posedirs = torch.cat([posedirs, torch.zeros_like(posedirs[:, :, :num_verts_teeth])], dim=2)  # (J, 9, V+num_verts_teeth, 3)
        self.posedirs = posedirs.reshape((len(self.parents) - 1) * 9, (num_verts_orig + num_verts_teeth) * 3)  # (J*9, (V+num_verts_teeth)*3)

        # J_regressor set to zero
        self.J_regressor = torch.cat([self.J_regressor, torch.zeros_like(self.J_regressor[:, :num_verts_teeth])], dim=1)  # (5, J) -> (5, J+num_verts_teeth)

        # lbs_weights manually set
        self.lbs_weights = torch.cat([self.lbs_weights, torch.zeros_like(self.lbs_weights[:num_verts_teeth])], dim=0)  # (V, 5) -> (V+num_verts_teeth, 5)
        self.lbs_weights[vid_teeth_upper, 1] += 1  # move with neck
        self.lbs_weights[vid_teeth_lower, 2] += 1  # move with jaw

        # add faces for teeth
        f_teeth_upper = torch.tensor([
            [0, 31, 30],  #0
            [0, 1, 31],  #1
            [1, 32, 31],  #2
            [1, 2, 32],  #3
            [2, 33, 32],  #4
            [2, 3, 33],  #5
            [3, 34, 33],  #6
            [3, 4, 34],  #7
            [4, 35, 34],  #8
            [4, 5, 35],  #9
            [5, 36, 35],  #10
            [5, 6, 36],  #11
            [6, 37, 36],  #12
            [6, 7, 37],  #13
            [7, 8, 37],  #14
            [8, 38, 37],  #15
            [8, 9, 38],  #16
            [9, 39, 38],  #17
            [9, 10, 39],  #18
            [10, 40, 39],  #19
            [10, 11, 40],  #20
            [11, 41, 40],  #21
            [11, 12, 41],  #22
            [12, 42, 41],  #23
            [12, 13, 42],  #24
            [13, 43, 42],  #25
            [13, 14, 43],  #26
            [14, 44, 43],  #27
        ], device=device, dtype=torch.int64)
        f_teeth_lower = torch.tensor([
            [45, 46, 15],  # 28           
            [46, 16, 15],  # 29
            [46, 47, 16],  # 30
            [47, 17, 16],  # 31
            [47, 48, 17],  # 32
            [48, 18, 17],  # 33
            [48, 49, 18],  # 34
            [49, 19, 18],  # 35
            [49, 50, 19],  # 36
            [50, 20, 19],  # 37
            [50, 51, 20],  # 38
            [51, 21, 20],  # 39
            [51, 52, 21],  # 40
            [52, 22, 21],  # 41
            [52, 23, 22],  # 42
            [52, 53, 23],  # 43
            [53, 24, 23],  # 44
            [53, 54, 24],  # 45
            [54, 25, 24],  # 46
            [54, 55, 25],  # 47
            [55, 26, 25],  # 48
            [55, 56, 26],  # 49
            [56, 27, 26],  # 50
            [56, 57, 27],  # 51
            [57, 28, 27],  # 52
            [57, 58, 28],  # 53
            [58, 29, 28],  # 54
            [58, 59, 29],  # 55
        ], device=device, dtype=torch.int64)
        self.faces = torch.cat([self.faces, f_teeth_upper + num_verts_orig, f_teeth_lower + num_verts_orig], dim=0)

        # add uv for teeth
        num_verts_uv_orig = self.uvs.shape[0]
        u = torch.linspace(0.75, 0.25, 15, device=device, dtype=dtype)
        v = torch.linspace(0.995, 0.955, 3, device=device, dtype=dtype)
        v = v[[0, 2, 1, 1]]
        uv = torch.stack(torch.meshgrid(u, v, indexing='ij'), dim=-1).permute(1, 0, 2).reshape(num_verts_teeth, 2)
        self.uvs = torch.cat([self.uvs, uv], dim=0)
        self.uv_faces = torch.cat([self.uv_faces, f_teeth_upper + num_verts_uv_orig, f_teeth_lower + num_verts_uv_orig], dim=0)


    def _vertices2landmarks(self, vertices, faces, lmk_faces_idx, lmk_bary_coords):
        """
            Calculates landmarks by barycentric interpolation
            Input:
                vertices: torch.tensor NxVx3, dtype = torch.float32
                    The tensor of input vertices
                faces: torch.tensor (N*F)x3, dtype = torch.long
                    The faces of the mesh
                lmk_faces_idx: torch.tensor N X L, dtype = torch.long
                    The tensor with the indices of the faces used to calculate the
                    landmarks.
                lmk_bary_coords: torch.tensor N X L X 3, dtype = torch.float32
                    The tensor of barycentric coordinates that are used to interpolate
                    the landmarks

            Returns:
                landmarks: torch.tensor NxLx3, dtype = torch.float32
                    The coordinates of the landmarks for each mesh in the batch
        """
        # Extract the indices of the vertices for each face
        # NxLx3
        batch_size, num_verts = vertices.shape[:2]
        device = vertices.device
        lmk_faces = torch.index_select(faces, 0, lmk_faces_idx.view(-1).to(torch.long)).view(batch_size, -1, 3)
        lmk_faces += torch.arange(batch_size, dtype=torch.long, device=device).view(-1, 1, 1) * num_verts
        lmk_vertices = vertices.view(-1, 3)[lmk_faces].view(batch_size, -1, 3, 3)
        landmarks = torch.einsum('blfi,blf->bli', [lmk_vertices, lmk_bary_coords])

        return landmarks

    def forward(self, 
        shape_params: torch.Tensor, 
        expression_params: torch.Tensor, 
        neck_pose_params: torch.Tensor,
        jaw_pose_params: torch.Tensor, 
        eye_pose_params: torch.Tensor, 
        eyelid_params: torch.Tensor
    ):
        batch_size = shape_params.shape[0]
        I = torch.eye(3, device=shape_params.device, dtype=self.v_template.dtype).unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)

        # Concatenate identity shape and expression parameters
        betas = torch.cat([shape_params, expression_params], dim=1)

        # The pose vector contains global rotation, and neck, jaw, and eyeball rotations
        # full_pose = torch.cat([rot_params, neck_pose_params, jaw_pose_params, eye_pose_params], dim=1)
        full_pose = torch.cat([
            I, 
            neck_pose_params.reshape(-1, 1, 3, 3),
            jaw_pose_params.reshape(-1, 1, 3, 3), 
            eye_pose_params.reshape(-1, 2, 3, 3)
        ], dim=1)

        # FLAME models shape and expression deformations as vertex offset from the mean face in 'zero pose', called v_template
        template_vertices = self.v_template.unsqueeze(0).expand(batch_size, -1, -1)

        # add eyelid before LBS
        if eyelid_params is not None:
            template_vertices = template_vertices + self.r_eyelid.expand(batch_size, -1, -1) * eyelid_params[:, 1:2, None]
            template_vertices = template_vertices + self.l_eyelid.expand(batch_size, -1, -1) * eyelid_params[:, 0:1, None]

        # Use linear blendskinning to model pose roations
        vertices, J = lbs(betas, full_pose, template_vertices,
                          self.shapedirs, self.posedirs,
                          self.J_regressor, self.parents,
                          self.lbs_weights, dtype=self.v_template.dtype,
                          pose2rot=False)

        # FIXME
        # if eyelid_params is not None:
        #     vertices = vertices + self.r_eyelid.expand(batch_size, -1, -1) * eyelid_params[:, 1:2, None]
        #     vertices = vertices + self.l_eyelid.expand(batch_size, -1, -1) * eyelid_params[:, 0:1, None]

        return vertices
    

    def compute_transfer(self,
        shape_params: torch.Tensor, 
        expression_params: torch.Tensor, 
        neck_pose_params: torch.Tensor,
        jaw_pose_params: torch.Tensor, 
        eye_pose_params: torch.Tensor, 
        eyelid_params: torch.Tensor
    ):
        batch_size = shape_params.shape[0]
        I = torch.eye(3, device=shape_params.device, dtype=self.v_template.dtype).unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)

        betas = torch.cat([shape_params, expression_params], dim=1)

        full_pose = torch.cat([
            I, 
            neck_pose_params.reshape(-1, 1, 3, 3),
            jaw_pose_params.reshape(-1, 1, 3, 3), 
            eye_pose_params.reshape(-1, 2, 3, 3)
        ], dim=1)

        template_vertices = self.v_template.unsqueeze(0).expand(batch_size, -1, -1)

        A = lbs_transfer(betas, full_pose, template_vertices,
            self.shapedirs, self.posedirs,
            self.J_regressor, self.parents,
            self.lbs_weights, dtype=self.v_template.dtype,
            pose2rot=False
        )
        return A


    def compute_keypoints(self, vertices: torch.Tensor):
        batch_size = vertices.shape[0]
        lmk_faces_idx = torch.cat([self.dynamic_lmk_faces_idx[0], self.lmk_faces_idx], 0)
        lmk_bary_coords = torch.cat([self.dynamic_lmk_bary_coords[0], self.lmk_bary_coords], 0)
        lmk_faces_idx = lmk_faces_idx.unsqueeze(dim=0).expand(batch_size, -1).contiguous()
        lmk_bary_coords = lmk_bary_coords.unsqueeze(dim=0).expand(batch_size, -1, -1).contiguous()
        lmk68 = self._vertices2landmarks(vertices, self.faces, lmk_faces_idx, lmk_bary_coords)
        return lmk68
    

    def compute_joint_location(self, shape_params: torch.Tensor, expression_params: torch.Tensor):
        betas = torch.cat([shape_params, expression_params], dim=1)
        template_vertices = self.v_template.unsqueeze(0).expand(betas.shape[0], -1, -1)
        v_shaped = template_vertices + blend_shapes(betas, self.shapedirs)
        J = vertices2joints(self.J_regressor, v_shaped)
        return J



