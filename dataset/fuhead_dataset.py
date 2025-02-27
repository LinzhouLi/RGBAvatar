import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from roma import rotmat_to_euler, unitquat_to_rotmat, quat_normalize
from submodules.fuhead import FuHead
from utils import rotation_6d_to_matrix


class FuHeadDataset(Dataset):
    def __init__(self, 
        fuhead_model: FuHead,
        path: str, 
        split: str = 'train', 
        use_shape_weight: bool = True,
        use_pose_weight: bool = True,
        preload: bool = False,
        pin_memory: bool = False
    ):
        self.preload = preload
        self.pin_memory = pin_memory
        image_folder_path = os.path.join(path, "images")
        fuhead_track_params_path = os.path.join(path, "fu_track_params.txt")
        fuhead_track_camera_path = os.path.join(path, "fu_track_camera.npz")

        self.image_paths = [os.path.join(image_folder_path, f) for f in sorted(os.listdir(image_folder_path))]
        track_params = np.loadtxt(fuhead_track_params_path, dtype=np.float32)
        first_frame_track_params = track_params[0]

        if split == 'train':
            self.image_paths = self.image_paths[:-350]
            track_params = track_params[:-350]
        elif split == 'test':
            self.image_paths = self.image_paths[-350:]
            track_params = track_params[-350:]
        elif split == 'all':
            pass
        else:
            raise ValueError("Invalid split")

        self.load_camera_params(fuhead_track_camera_path, first_frame_track_params)
        self.load_track_params(track_params)
        self.precompute_bind_mesh(fuhead_model)
        self.precompute_blend_weight(use_shape_weight, use_pose_weight)
        if preload: self.load_images()
        else: self.images = None

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        if self.images is None:
            image_np = np.array(Image.open(self.image_paths[index]))
            image_tf = torch.from_numpy(image_np).permute(2, 0, 1)
        else:
            image_tf = self.images[index]
        return {
            'image': image_tf,
            'mesh': self.mesh_verts[index],
            'blend_weight': self.blend_weight[index]
        }
    
    def load_images(self):
        images = []
        for img_path in tqdm(self.image_paths, desc="Loading images"):
            img = np.array(Image.open(img_path))
            images.append(img)
        self.images = torch.from_numpy(np.stack(images, axis=0)).permute(0, 3, 1, 2).contiguous() # caution: pre contiguous here
        if self.pin_memory: self.images = self.images.pin_memory() # use pinned memory
        print("Load images")

    def load_camera_params(self, camera_data_path: str, first_frame_track_params: np.ndarray):
        camera_data = np.load(camera_data_path)
        self.camera_intri = camera_data['intrinsics']
        self.camera_extri = np.diag([1.0, -1.0, -1.0, 1.0]) # adjust the camera coordinate
        translation = first_frame_track_params[55:58].copy() # use the first frame camera pose # !!!caution: copy here to prevent in-place modification
        translation[1:3] = -translation[1:3]
        self.camera_extri[:3,  3] = translation
        self.image_width = camera_data["image_width"].item()
        self.image_height = camera_data["image_height"].item()
        print("Load camera params with frame 0")

    def load_track_params(self, track_params: np.ndarray):
        self.expressons = torch.from_numpy(track_params[:, :51])
        self.identity = torch.from_numpy(track_params[-1, -231:]) # use identity param in the last frame
        self.eye_rots = torch.from_numpy(track_params[:, 58:62])
        self.eye_rots = unitquat_to_rotmat(quat_normalize(self.eye_rots))

        global_rots = torch.from_numpy(track_params[:, 51:55])
        global_rots = unitquat_to_rotmat(quat_normalize(global_rots))
        global_transls = torch.from_numpy(track_params[:, 55:58])
        global_rots[:, 1:3] = -global_rots[:, 1:3] # adjust camera coordinate
        global_transls[:, 1:3] = -global_transls[:, 1:3]
        transform_mat = torch.eye(4).unsqueeze(0).repeat(len(global_rots), 1, 1)
        transform_mat[:, :3, :3] = global_rots
        transform_mat[:, :3, 3] = global_transls

        transform_mat = transform_mat.to(dtype=torch.float64)
        inv_cam_extri = torch.from_numpy(np.linalg.inv(self.camera_extri))
        transform_mat = torch.matmul(inv_cam_extri.unsqueeze(0), transform_mat)
        transform_mat = transform_mat.to(dtype=torch.float32)
        self.global_rots = transform_mat[:, :3, :3]
        self.global_transls = transform_mat[:, :3, 3]
        print("Load FuHead params")

    def precompute_bind_mesh(self, fuhead_model: FuHead):
        verts = fuhead_model(self.identity.cuda(), self.expressons.cuda(), self.eye_rots.cuda())
        verts = torch.matmul(verts, self.global_rots.transpose(-1, -2).cuda())
        verts += self.global_transls.unsqueeze(1).cuda()
        self.mesh_verts = verts.to(dtype=torch.float32, device='cpu')

    def precompute_blend_weight(self, use_shape_weight: bool, use_pose_weight: bool):
        shape_weight = self.expressons
        pose_weight = torch.cat([
            self.eye_rots[:, :2, :].reshape(-1, 6), # matrix to rot6d
            self.global_rots[:, :2, :].reshape(-1, 6), # matrix to rot6d
            self.global_transls
        ], dim=1)
        blend_weights = []
        if use_shape_weight: blend_weights.append(shape_weight)
        if use_pose_weight: blend_weights.append(pose_weight)
        self.blend_weight = torch.cat(blend_weights, dim=1).to(dtype=torch.float32, device='cpu')