from typing import Optional

import torch
from torch import nn
from torch.nn import Parameter
import numpy as np
from plyfile import PlyData, PlyElement

from diff_renderer import GaussianAttributes
from diff_gaussian_rasterization import linear_blending
from utils import Struct, flatten_model_params, load_flattened_model_params, inverse_sigmoid

# legacy
def linear_blending_torch(basis: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    num_batch, num_bias = weight.shape
    data_dim = basis.ndim
    result = basis.unsqueeze(0) + (weight.view(num_batch, num_bias, *([1] * data_dim)) * bias.unsqueeze(0)).sum(dim=1)
    return result


class GaussianModel:
    def __init__(self, model_config: Struct):
        self.model_config = model_config
        self._xyz = torch.empty(0)
        self._opacity = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._feature_dc = torch.empty(0)

        self._xyz_b = torch.empty(0)
        self._feature_b = torch.empty(0)
        self._rotation_b = torch.empty(0)

        if self.model_config.use_mlp_proj:
            self.weight_module = nn.Sequential( # MLP
                nn.Linear(self.model_config.num_basis_in, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, self.model_config.num_basis_blend)
            ).cuda()
        else:
            self.weight_module = nn.Linear(self.model_config.num_basis_in, self.model_config.num_basis_blend, device='cuda')

        self.opacity_act = torch.sigmoid
        self.inv_opactity_act = inverse_sigmoid
        self.scaling_act = torch.exp
        self.inv_scaling_act = torch.log
        self.rotation_act = lambda x:  torch.nn.functional.normalize(x, dim=-1)
    
    def project_weight(self, blend_weight_in: torch.Tensor):
        blend_weight_in = blend_weight_in[:, :self.model_config.num_basis_in]
        if self.model_config.use_weight_proj:
            blend_weight_proj = self.weight_module(blend_weight_in)
            return blend_weight_proj
        else:
            return blend_weight_in

    # legacy
    def get_attributes_torch(self, blend_weight: Optional[torch.Tensor] = None):
        if blend_weight is not None and self.model_config.use_blend:
            blend_weight = self.project_weight(blend_weight.unsqueeze(0)) # add batch dim
            _xyz = linear_blending_torch(self._xyz, blend_weight, self._xyz_b).squeeze(0)
            _rotation = linear_blending_torch(self._rotation, blend_weight, self._rotation_b).squeeze(0)
            _feature_dc = linear_blending_torch(self._feature_dc, blend_weight, self._feature_b).squeeze(0)
            _opacity, _scaling = self._opacity, self._scaling # do not blend opacity and scaling
        else:
            _xyz, _rotation, _opacity, _scaling, _feature_dc = self._xyz, self._rotation, self._opacity, self._scaling, self._feature_dc
        return GaussianAttributes(
            _xyz, 
            self.opacity_act(_opacity), 
            self.scaling_act(_scaling), 
            self.rotation_act(_rotation), 
            _feature_dc
        )
    
    # legacy
    def get_batch_attributes_torch(self, batch_size: int, blend_weight: Optional[torch.Tensor] = None):
        if blend_weight is not None and self.model_config.use_blend:
            blend_weight = self.project_weight(blend_weight)
            _xyz = linear_blending_torch(self._xyz, blend_weight, self._xyz_b)
            _rotation = linear_blending_torch(self._rotation, blend_weight, self._rotation_b)
            _feature_dc = linear_blending_torch(self._feature_dc, blend_weight, self._feature_b)
            _opacity = self._opacity.expand(batch_size, -1, -1)
            _scaling = self._scaling.expand(batch_size, -1, -1)
        else:
            _xyz = self._xyz.expand(batch_size, -1, -1)
            _rotation = self._rotation.expand(batch_size, -1, -1)
            _opacity = self._opacity.expand(batch_size, -1, -1)
            _scaling = self._scaling.expand(batch_size, -1, -1)
            _feature_dc = self._feature_dc.expand(batch_size, -1, -1, -1)
        return GaussianAttributes(
            _xyz, 
            self.opacity_act(_opacity), 
            self.scaling_act(_scaling), 
            self.rotation_act(_rotation), 
            _feature_dc
        )
    
    def get_attributes(self, blend_weight: Optional[torch.Tensor] = None):
        if blend_weight is not None and self.model_config.use_blend:
            blend_weight = self.project_weight(blend_weight.unsqueeze(0)) # add batch dim
            _xyz , _rotation, _feature_dc = linear_blending( # NOTE: if blend_weight is unsqueezed one more time, it will cause strange rendering results
                blend_weight,
                self._xyz, self._rotation, self._feature_dc,
                self._xyz_b, self._rotation_b, self._feature_b
            )
            _xyz , _rotation, _feature_dc = _xyz.squeeze(0), _rotation.squeeze(0), _feature_dc.squeeze(0)
            _opacity, _scaling = self._opacity, self._scaling # do not blend opacity and scaling
        else:
            _xyz, _rotation, _opacity, _scaling, _feature_dc = self._xyz, self._rotation, self._opacity, self._scaling, self._feature_dc
        return GaussianAttributes(
            _xyz, 
            self.opacity_act(_opacity), 
            self.scaling_act(_scaling), 
            self.rotation_act(_rotation), 
            _feature_dc
        )
    
    def get_batch_attributes(self, batch_size: int, blend_weight: Optional[torch.Tensor] = None):
        if blend_weight is not None and self.model_config.use_blend:
            blend_weight = self.project_weight(blend_weight)
            _xyz, _rotation, _feature_dc = linear_blending(
                blend_weight,
                self._xyz, self._rotation, self._feature_dc,
                self._xyz_b, self._rotation_b, self._feature_b
            )
            _opacity = self._opacity.expand(batch_size, -1, -1)
            _scaling = self._scaling.expand(batch_size, -1, -1)
        else:
            _xyz = self._xyz.expand(batch_size, -1, -1)
            _rotation = self._rotation.expand(batch_size, -1, -1)
            _opacity = self._opacity.expand(batch_size, -1, -1)
            _scaling = self._scaling.expand(batch_size, -1, -1)
            _feature_dc = self._feature_dc.expand(batch_size, -1, -1, -1)
        return GaussianAttributes(
            _xyz, 
            self.opacity_act(_opacity), 
            self.scaling_act(_scaling), 
            self.rotation_act(_rotation), 
            _feature_dc
        )
    
    def sparsity_loss(self, blend_weight: torch.Tensor):
        if blend_weight is not None and self.model_config.use_blend and self.model_config.use_weight_proj:
            return torch.abs(self.weight_module(blend_weight)).mean()
        else:
            return 0.0

    def training_params(self, args):
        gs_params = [
            {'params': [self._xyz], 'lr': args.position_lr * args.scene_extent, "name": "xyz"},
            {'params': [self._opacity], 'lr': args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': args.rotation_lr, "name": "rotation"},
            {'params': [self._feature_dc], 'lr': args.feature_lr, "name": "f_dc"}
        ]
        bs_params = [
            {'params': [self._xyz_b], 'lr': args.position_b_lr_scale * args.position_lr * args.scene_extent, "name": "xyz_b"},
            {'params': [self._rotation_b], 'lr': args.rotaion_b_lr_scale * args.rotation_lr, "name": "rotation_b"},
            {'params': [self._feature_b], 'lr': args.feature_b_lr_scale * args.feature_lr, "name": "f_dc_b"}
        ]
        adapter_params = [
            {'params': self.weight_module.parameters(), 'lr': args.weight_module_lr, "name": "weight_module"}
        ]
        return gs_params, bs_params, adapter_params

    @torch.no_grad()
    def save_ply(self, path: str):
        xyz = self._xyz.cpu().numpy()
        num_gs = xyz.shape[0]
        normal = np.zeros_like(xyz)
        opacity = self._opacity.cpu().numpy()
        scaling = self._scaling.cpu().numpy()
        rotation = self._rotation.cpu().numpy()
        f_dc = self._feature_dc.cpu().transpose(1, 2).flatten(start_dim=1).numpy()
        f_rest = np.zeros([num_gs, 45], dtype=xyz.dtype)

        xyz_b = self._xyz_b.transpose(0, 1).reshape([num_gs, -1]).contiguous().cpu().numpy()
        rotation_b = self._rotation_b.transpose(0, 1).reshape([num_gs, -1]).contiguous().cpu().numpy()
        f_dc_b = self._feature_b.transpose(0, 1).reshape([num_gs, -1]).contiguous().cpu().numpy()

        linear_module = flatten_model_params(self.weight_module).cpu().numpy().reshape([-1, 1])
        assert linear_module.shape[0] <= num_gs
        linear_module_save = np.zeros([num_gs, 1], dtype=linear_module.dtype)
        linear_module_save[:linear_module.shape[0]] = linear_module

        binding = self.binding_face_id is not None and self.binding_face_bary is not None
        if binding:
            binding_face_id = self.binding_face_id.cpu().numpy().astype(np.int32)[..., np.newaxis]
            binding_face_bary = self.binding_face_bary.cpu().numpy()

        l = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'opacity']
        for i in range(scaling.shape[1]): l.append('scale_{}'.format(i))
        for i in range(rotation.shape[1]): l.append('rot_{}'.format(i))
        for i in range(f_dc.shape[1]): l.append('f_dc_{}'.format(i))
        for i in range(f_rest.shape[1]): l.append('f_rest_{}'.format(i))
        for i in range(xyz_b.shape[1]): l.append('xyz_b_{}'.format(i))
        for i in range(rotation_b.shape[1]): l.append('rot_b_{}'.format(i))
        for i in range(f_dc_b.shape[1]): l.append('f_dc_b_{}'.format(i))
        l.append('weight_module')
        dtype_full = [(attribute, 'f4') for attribute in l]

        if binding:
            dtype_full.append(('face_id', 'i4'))
            dtype_full.append(('face_bary_0', 'f4'))
            dtype_full.append(('face_bary_1', 'f4'))
            dtype_full.append(('face_bary_2', 'f4'))

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normal, opacity, scaling, rotation, f_dc, f_rest, xyz_b, rotation_b, f_dc_b, linear_module_save), axis=1)

        if binding:
            attributes = np.concatenate((attributes, binding_face_id, binding_face_bary), axis=1)

        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    @torch.no_grad()
    def load_ply(self, path: str, train: bool = False):
        plydata = PlyData.read(path)

        xyz = np.stack((
            np.asarray(plydata.elements[0]["x"], dtype=np.float32),
            np.asarray(plydata.elements[0]["y"], dtype=np.float32),
            np.asarray(plydata.elements[0]["z"], dtype=np.float32)
        ), axis=1)
        num_gaussian = xyz.shape[0]

        opacity = np.asarray(plydata.elements[0]["opacity"], dtype=np.float32)[..., np.newaxis]
        assert opacity.shape[0] == num_gaussian

        scaling_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scaling_names = sorted(scaling_names, key = lambda x: int(x.split('_')[-1]))
        scaling = np.zeros((num_gaussian, len(scaling_names)), dtype=np.float32)
        for idx, attr_name in enumerate(scaling_names):
            scaling[:, idx] = np.asarray(plydata.elements[0][attr_name], dtype=np.float32)
        assert scaling.shape[0] == num_gaussian

        rotation = np.stack((
            np.asarray(plydata.elements[0]["rot_0"], dtype=np.float32),
            np.asarray(plydata.elements[0]["rot_1"], dtype=np.float32),
            np.asarray(plydata.elements[0]["rot_2"], dtype=np.float32),
            np.asarray(plydata.elements[0]["rot_3"], dtype=np.float32)
        ), axis=1)
        assert rotation.shape[0] == num_gaussian

        feature_dc = np.stack((
            np.asarray(plydata.elements[0]["f_dc_0"], dtype=np.float32),
            np.asarray(plydata.elements[0]["f_dc_1"], dtype=np.float32),
            np.asarray(plydata.elements[0]["f_dc_2"], dtype=np.float32)
        ), axis=1).reshape([num_gaussian, 3, 1])
        assert feature_dc.shape[0] == num_gaussian

        xyz_b = np.stack([
            np.asarray(plydata.elements[0]["xyz_b_{}".format(i)], dtype=np.float32)
            for i in range(self.model_config.num_basis_blend * 3)
        ], axis=1)
        assert xyz_b.shape[0] == num_gaussian
        xyz_b = xyz_b.reshape([num_gaussian, self.model_config.num_basis_blend, 3])

        rotation_b = np.stack([
            np.asarray(plydata.elements[0]["rot_b_{}".format(i)], dtype=np.float32)
            for i in range(self.model_config.num_basis_blend * 4)
        ], axis=1)
        assert rotation_b.shape[0] == num_gaussian
        rotation_b = rotation_b.reshape([num_gaussian, self.model_config.num_basis_blend, 4])

        f_dc_b = np.stack([
            np.asarray(plydata.elements[0]["f_dc_b_{}".format(i)], dtype=np.float32)
            for i in range(self.model_config.num_basis_blend * 3)
        ], axis=1)
        assert f_dc_b.shape[0] == num_gaussian
        f_dc_b = f_dc_b.reshape([num_gaussian, self.model_config.num_basis_blend, 1, 3])

        linear_module = np.asarray(plydata.elements[0]["weight_module"], dtype=np.float32)
        linear_module = torch.from_numpy(linear_module).cuda()
        load_flattened_model_params(linear_module, self.weight_module)

        p_names = [p.name for p in plydata.elements[0].properties]
        if 'face_id' in p_names:
            binding_face_id = np.asarray(plydata.elements[0]['face_id'], dtype=np.int32)
            assert binding_face_id.shape[0] == num_gaussian
            self.binding_face_id = torch.from_numpy(binding_face_id).cuda()
            print("Binding face id loaded")
        
        bary_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("face_bary_")]
        if len(bary_names) > 0:
            bary_names = sorted(bary_names, key = lambda x: int(x.split('_')[-1]))
            binding_face_bary = np.zeros((num_gaussian, len(bary_names)), dtype=np.float32)
            for idx, attr_name in enumerate(bary_names):
                binding_face_bary[:, idx] = np.asarray(plydata.elements[0][attr_name], dtype=np.float32)
            assert binding_face_bary.shape[0] == num_gaussian
            self.binding_face_bary = torch.from_numpy(binding_face_bary).cuda()
            print("Binding face barycentric loaded")

        self._xyz = torch.from_numpy(xyz).cuda()
        self._opacity = torch.from_numpy(opacity).cuda()
        self._scaling = torch.from_numpy(scaling).cuda()
        self._rotation = torch.from_numpy(rotation).cuda()
        self._feature_dc = torch.from_numpy(feature_dc).transpose(1, 2).contiguous().cuda()

        self._xyz_b = torch.from_numpy(xyz_b).transpose(0, 1).contiguous().cuda()
        self._rotation_b = torch.from_numpy(rotation_b).transpose(0, 1).contiguous().cuda()
        self._feature_b = torch.from_numpy(f_dc_b).transpose(0, 1).contiguous().cuda()

        if train:
            self._xyz = Parameter(self._xyz.requires_grad_(True))
            self._opacity = Parameter(self._opacity.requires_grad_(True))
            self._scaling = Parameter(self._scaling.requires_grad_(True))
            self._rotation = Parameter(self._rotation.requires_grad_(True))
            self._feature_dc = Parameter(self._feature_dc.requires_grad_(True))

            self._xyz_b = Parameter(self._xyz_b.requires_grad_(True))
            self._rotation_b = Parameter(self._rotation_b.requires_grad_(True))
            self._feature_b = Parameter(self._feature_b.requires_grad_(True))

    @torch.no_grad()
    def load_weight_module(self, path: str):
        plydata = PlyData.read(path)
        linear_module = np.asarray(plydata.elements[0]["weight_module"], dtype=np.float32)
        linear_module = torch.from_numpy(linear_module).cuda()
        load_flattened_model_params(linear_module, self.weight_module)
    
    @torch.no_grad()
    def capture(self):
        return (
            self._xyz,
            self._opacity,
            self._scaling,
            self._rotation,
            self._feature_dc,
            self._xyz_b,
            self._rotation_b,
            self._feature_b,
            self.weight_module.weight,
            self.weight_module.bias
        )
    
    @torch.no_grad()
    def restore(self, params):
        self._xyz.copy_(params[0])
        self._opacity.copy_(params[1])
        self._scaling.copy_(params[2])
        self._rotation.copy_(params[3])
        self._feature_dc.copy_(params[4])
        self._xyz_b.copy_(params[5])
        self._rotation_b.copy_(params[6])   
        self._feature_b.copy_(params[7])
        self.weight_module.weight.copy_(params[8])
        self.weight_module.bias.copy_(params[9])

    @torch.no_grad()
    def restore_from_optimizer(self, optimizer: torch.optim.Optimizer):
        for group in optimizer.param_groups:
            if group['name'] == 'xyz': self._xyz = group['params'][0]
            elif group['name'] == 'opacity': self._opacity = group['params'][0]
            elif group['name'] == 'scaling': self._scaling = group['params'][0]
            elif group['name'] == 'rotation': self._rotation = group['params'][0]
            elif group['name'] == 'f_dc': self._feature_dc = group['params'][0]
            elif group['name'] == 'xyz_b': self._xyz_b = group['params'][0]
            elif group['name'] == 'rotation_b': self._rotation_b = group['params'][0]
            elif group['name'] == 'f_dc_b': self._feature_b = group['params'][0]
            elif group['name'] == 'weight_module':
                self.weight_module.weight = group['params'][0]
                self.weight_module.bias = group['params'][1]