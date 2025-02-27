import os
import struct
import numpy as np
import torch
from torch import nn


def read_fuhead_tensor(config):
    # m_mean_shape
    file = open(config.mean_shape_tensor_path, 'rb')
    bytes_read_n = file.read(4)
    val_n = struct.unpack('i', bytes_read_n)
    length =  val_n[0] * 3
    bytes_read_tensor = file.read(4 * length)
    m_mean_shape = struct.unpack(length * 'f', bytes_read_tensor)
    m_mean_shape = np.array(m_mean_shape, dtype=np.float32)
    file.close()

    # m_id_tensor
    file = open(config.identity_tensor_path, 'rb')
    bytes_read_n = file.read(4)
    bytes_read_id = file.read(4)
    val_n = struct.unpack('i', bytes_read_n)
    val_id = struct.unpack('i', bytes_read_id)
    length =  val_n[0] * val_id[0] * 3
    bytes_read_tensor = file.read(4 * length)
    m_id_tensor = struct.unpack(length * 'f', bytes_read_tensor)
    m_id_tensor = np.array(m_id_tensor, dtype=np.float32)
    m_id_tensor = np.reshape(m_id_tensor, [val_id[0],val_n[0] * 3])
    m_id_tensor = np.transpose(m_id_tensor, [1, 0])
    file.close()

    # m_expr_tensor
    file = open(config.expr_tensor_path, 'rb')
    bytes_read_n = file.read(4)
    bytes_read_exp = file.read(4)
    val_n = struct.unpack('i', bytes_read_n)
    val_exp= struct.unpack('i', bytes_read_exp)
    length =  val_n[0] * 3
    bytes_read_tensor = file.read(4 * length)
    length =  val_n[0] * val_exp[0] * 3
    bytes_read_tensor = file.read(4 * length)
    length =  val_n[0] * val_exp[0] * 3
    bytes_read_tensor = file.read(4 * length)
    m_expr_tensor = struct.unpack(length * 'f', bytes_read_tensor)
    m_expr_tensor = np.array(m_expr_tensor, dtype=np.float32)
    m_expr_tensor = np.reshape(m_expr_tensor, [val_exp[0], val_n[0] * 3])
    m_expr_tensor = np.transpose(m_expr_tensor, [1, 0])
    file.close()

    # m_transfered_iden_tensor
    file = open(config.transfer_tensor_path, 'rb')
    length =  2504 * 3 * 51 * 231
    bytes_read_tensor = file.read(4 * length)
    m_transfered_iden_tensor = struct.unpack(length * 'f', bytes_read_tensor)
    m_transfered_iden_tensor = np.array(m_transfered_iden_tensor, dtype=np.float32)
    m_transfered_iden_tensor = np.reshape(m_transfered_iden_tensor, [2504 * 3, 51, 231])
    file.close()


class FuHead(nn.Module):
    def __init__(self, data_path: str = "./data/fu_head/fu_head.npz"):
        super(FuHead, self).__init__()
        print(f"[FuHead] Creating the 3DMM from {data_path}")

        data = np.load(data_path)

        # add teeth
        teeth_upper_v, teeth_lower_v, teeth_faces, teeth_uvs = self.add_teeth(data["head_v"])

        # stack vertices
        v_template = np.concatenate([data["head_v"], data["left_eyeball_v"], data["right_eyeball_v"], teeth_upper_v, teeth_lower_v])

        # stack uvs
        uvs = np.concatenate([data["head_uv"], data["left_eyeball_uv"], data["right_eyeball_uv"], teeth_uvs])
        
        # stack vertice faces
        faces = np.concatenate([
            data["head_faces"],
            data["left_eyeball_faces"] + len(data["head_v"]),
            data["right_eyeball_faces"] + len(data["head_v"]) + len(data["left_eyeball_v"]),
            teeth_faces + len(data["head_v"]) + len(data["left_eyeball_v"]) + len(data["right_eyeball_v"])
        ])

        # stack uv faces
        uv_faces = np.concatenate([
            data["head_uv_faces"],
            data["left_eyeball_uv_faces"] + len(data["head_uv"]),
            data["right_eyeball_uv_faces"] + len(data["head_uv"]) + len(data["left_eyeball_uv"]),
            teeth_faces + len(data["head_uv"]) + len(data["left_eyeball_uv"]) + len(data["right_eyeball_uv"])
        ])

        # hard coded
        left_eye_transl = np.array([-0.032, 0.029, 0.027], dtype=np.float32)
        right_eye_transl = np.array([0.032, 0.029, 0.027], dtype=np.float32)

        self.num_vertices = len(data["head_v"]) + len(data["left_eyeball_v"]) + len(data["right_eyeball_v"]) + len(teeth_upper_v) + len(teeth_lower_v)
        self.num_faces = len(faces)
        self.num_blend_basis = 63

        # register buffer
        self.register_buffer("m_mean_shape", torch.from_numpy(data["m_mean_shape"]))
        self.register_buffer("m_id_tensor", torch.from_numpy(data["m_id_tensor"]))
        self.register_buffer("m_expr_tensor", torch.from_numpy(data["m_expr_tensor"]))
        self.register_buffer("m_transfered_iden_tensor", torch.from_numpy(data["m_transfered_iden_tensor"]))

        left_eyeball_verts = data["left_eyeball_v"] * 1.2 # scale, hard coded
        right_eyeball_verts = data["right_eyeball_v"] * 1.2
        self.register_buffer("left_eyeball_base_verts", torch.from_numpy(left_eyeball_verts))
        self.register_buffer("right_eyeball_base_verts", torch.from_numpy(right_eyeball_verts))
        
        self.register_buffer("teeth_upper_verts", torch.from_numpy(teeth_upper_v))
        self.register_buffer("teeth_lower_verts", torch.from_numpy(teeth_lower_v))

        self.register_buffer("left_eye_transl", torch.from_numpy(left_eye_transl))
        self.register_buffer("right_eye_transl", torch.from_numpy(right_eye_transl))

        self.register_buffer("v_template", torch.from_numpy(v_template))
        self.register_buffer("faces", torch.from_numpy(faces))
        self.register_buffer("uvs", torch.from_numpy(uvs))
        self.register_buffer("uv_faces", torch.from_numpy(uv_faces))
        print("Done!")


    def add_teeth(self, head_v):
        head_v = torch.from_numpy(head_v)
        device = head_v.device
        dtype = head_v.dtype

        vid_lip_outside_ring_upper = torch.tensor([
            685, 686, 687, 688, 689, 690, 691, 24, 256, 255, 254, 253, 252, 251, 250
        ], device=device, dtype=torch.int64)
        vid_lip_outside_ring_lower = torch.tensor([
            834, 683, 682, 710, 725, 709, 700, 25, 265, 274, 290, 275, 247, 248, 404
        ], device=device, dtype=torch.int64)

        v_lip_upper = head_v[vid_lip_outside_ring_upper]
        v_lip_lower = head_v[vid_lip_outside_ring_lower]

        # construct vertices for teeth
        mean_dist = 0.008
        v_teeth_middle = (v_lip_upper + v_lip_lower) / 2
        v_teeth_middle[:, 1] = v_teeth_middle[:, 1:2].mean(dim=0, keepdim=True)
        v_teeth_middle[:, 2] -= 0.004  # how far the teeth are from the lips

        # upper, front
        v_teeth_upper_edge = v_teeth_middle.clone() + torch.tensor([[0, mean_dist, 0]], device=device, dtype=dtype) * 1.0
        v_teeth_upper_root = v_teeth_upper_edge - torch.tensor([[0, mean_dist, 0]], device=device, dtype=dtype) * 1.2  # scale the height of teeth

        # lower, front
        v_teeth_lower_edge = v_teeth_middle.clone() - torch.tensor([[0, mean_dist, 0]], device=device, dtype=dtype) * 1.0
        v_teeth_lower_edge -= torch.tensor([[0, 0, mean_dist]], device=device, dtype=dtype) * 0.4  # slightly move the lower teeth to the back
        v_teeth_lower_root = v_teeth_lower_edge + torch.tensor([[0, mean_dist, 0]], device=device, dtype=dtype) * 1.0  # scale the height of teeth

        # vertices
        v_teeth_upper = torch.cat([
            v_teeth_upper_edge,
            v_teeth_upper_root
        ], dim=0)
        v_teeth_lower = torch.cat([
            v_teeth_lower_edge,
            v_teeth_lower_root
        ], dim=0)
        num_verts_teeth = v_teeth_upper.shape[0] + v_teeth_lower.shape[0]

        # faces
        f_teeth_upper = torch.tensor([
            [0, 16, 15],
            [0, 1, 16]
        ], device=device, dtype=torch.int64)
        f_teeth_upper = torch.cat([f_teeth_upper + i for i in range(14)])
        f_teeth_lower = torch.tensor([
            [45, 46, 30],       
            [46, 31, 30]
        ], device=device, dtype=torch.int64)
        f_teeth_lower = torch.cat([f_teeth_lower + i for i in range(14)])
        faces = torch.cat([f_teeth_upper, f_teeth_lower])

        # uvs
        u = torch.linspace(0.7, 0.3, 15, device=device, dtype=dtype)
        v = torch.linspace(0.995, 0.90, 3, device=device, dtype=dtype)
        v = v[[1, 0, 1, 2]]
        uvs = torch.stack(torch.meshgrid(u, v, indexing='ij'), dim=-1).permute(1, 0, 2).reshape(num_verts_teeth, 2)
        return v_teeth_upper.numpy(), v_teeth_lower.numpy(), faces.numpy(), uvs.numpy()


    def forward_identity(self, identity: torch.Tensor):
        head_base_verts = self.m_mean_shape + torch.matmul(self.m_id_tensor, identity) # [7512]
        transfer_tensor = self.m_expr_tensor + torch.tensordot(self.m_transfered_iden_tensor, identity, dims=([2], [0])) # [7512, 51]
        return head_base_verts, transfer_tensor


    def forward_expression(self, 
        head_base_verts: torch.Tensor,
        transfer_tensor: torch.Tensor,
        expressions: torch.Tensor, 
        eye_rots: torch.Tensor
    ) -> torch.Tensor:
        batch_size = expressions.shape[0]
        delta_verts = torch.matmul(expressions, transfer_tensor.T)
        head_verts = head_base_verts + delta_verts
        head_verts = head_verts.reshape(batch_size, -1, 3) # [B, 2504, 3]
        left_eyeball_verts = torch.matmul(self.left_eyeball_base_verts, eye_rots.transpose(-1, -2)) + self.left_eye_transl
        right_eyeball_verts = torch.matmul(self.right_eyeball_base_verts, eye_rots.transpose(-1, -2)) + self.right_eye_transl
        teeth_upper_verts = self.teeth_upper_verts.unsqueeze(0).repeat(batch_size, 1, 1)
        teeth_lower_verts = self.teeth_lower_verts.unsqueeze(0).repeat(batch_size, 1, 1)
        teeth_lower_verts += delta_verts[:, 102:105].unsqueeze(1)
        vertices = torch.cat([head_verts, left_eyeball_verts, right_eyeball_verts, teeth_upper_verts, teeth_lower_verts], dim=1)
        return vertices

    
    def forward(self,
        identity: torch.Tensor, # no batch dim, shape: [231]
        expressions: torch.Tensor, # with batch dim, shape: [B, 51]
        eye_rots: torch.Tensor
    ) -> torch.Tensor:
        head_base_verts, transfer_tensor = self.forward_identity(identity)
        vertices = self.forward_expression(head_base_verts, transfer_tensor, expressions, eye_rots)
        return vertices