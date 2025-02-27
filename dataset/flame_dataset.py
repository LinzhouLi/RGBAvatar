import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from roma import rotmat_to_euler
from submodules.flame import FLAME
from utils import rotation_6d_to_matrix


class FLAMEDataset(Dataset):
    def __init__(self, 
        flame_model: FLAME,
        path: str, 
        split: str = 'train', 
        use_shape_weight: bool = True,
        use_pose_weight: bool = True,
        preload: bool = False,
        pin_memory: bool = False,
        retarget_neck: bool = True,
        HQ: bool = False
    ):
        self.preload = preload
        self.pin_memory = pin_memory
        self.HQ = HQ
        if self.HQ: image_folder_path = os.path.join(path, "images.HQ")
        else: image_folder_path = os.path.join(path, "images")
        # image_folder_path = os.path.join(path, "matted")
        flame_folder_path = os.path.join(path, "checkpoint")

        self.image_paths = [os.path.join(image_folder_path, f) for f in sorted(os.listdir(image_folder_path))]
        self.flame_param_paths = [os.path.join(flame_folder_path, f) for f in sorted(os.listdir(flame_folder_path))]

        if split == 'train':
            self.image_paths = self.image_paths[:-350]
            self.flame_param_paths = self.flame_param_paths[:-350]
        elif split == 'test':
            self.image_paths = self.image_paths[-350:]
            self.flame_param_paths = self.flame_param_paths[-350:]
        elif split == 'all':
            pass
        else:
            raise ValueError("Invalid split")

        self.load_camera_params(flame_folder_path) # caution: use frame 0 camera pose
        self.load_track_params()
        if retarget_neck:
            self.transform_flame_params(flame_model) # retarget neck rotation to global rotation
        self.precompute_bind_mesh(flame_model)
        self.precompute_blend_weight(use_shape_weight, use_pose_weight)
        if preload: self.load_images()
        else: self.images = None

    def __len__(self):
        return len(self.image_paths)

    def get_image(self, index):
        if self.images is None:
            image_np = np.array(Image.open(self.image_paths[index]))
        else:
            image_tf = self.images[index]
            image_np = image_tf.permute(1, 2, 0).numpy()
        return image_np
    
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
    
    def get_roll_yaw_pitch(self, index):
        global_rots = self.neck_poses if self.pose_transformed else self.global_rots
        global_rot = rotmat_to_euler("zyx", global_rots[index], degrees=True)
        roll, yaw, pitch = global_rot[0], global_rot[1], global_rot[2]
        return roll, yaw, pitch
    
    def load_images(self):
        images = []
        for img_path in tqdm(self.image_paths, desc="Loading images"):
            img = np.array(Image.open(img_path))
            images.append(img)
        self.images = torch.from_numpy(np.stack(images, axis=0)).permute(0, 3, 1, 2).contiguous() # caution: pre contiguous here
        if self.pin_memory: self.images = self.images.pin_memory() # use pinned memory
        print("Load images")

    def load_camera_params(self, path: str):
        flame_data = torch.load(os.path.join(path, "00000.frame"))
        self.camera_intri = flame_data['opencv']['K'][0]
        self.camera_extri = np.diag([1.0, -1.0, -1.0, 1.0]) # adjust the camera coordinate
        self.camera_extri[:3,  3] = flame_data['opencv']['t'][0]
        self.image_width = flame_data['img_size'][0]
        self.image_height = flame_data['img_size'][1]
        print("Load camera params with frame 0")
        if self.HQ:
            self.camera_intri[0:2] *= 2
            self.image_width *= 2
            self.image_height *= 2

    def load_track_params(self):
        shapes = []
        exps = []
        jaw_poses = []
        eye_poses = []
        eyelid_params = []
        global_rots = []
        global_transls = []

        inv_cam_extri = np.linalg.inv(self.camera_extri)
        for flame_param_path in tqdm(self.flame_param_paths, desc="Loading flame params"):
            flame_data = torch.load(flame_param_path)
            jaw_pose = torch.from_numpy(flame_data['flame']['jaw']).to(torch.float64)
            eye_pose = torch.from_numpy(flame_data['flame']['eyes']).to(torch.float64)
            eyelid_param = torch.from_numpy(flame_data['flame']['eyelids']).to(torch.float64)
            shape = torch.from_numpy(flame_data['flame']['shape']).to(torch.float64)
            exp = torch.from_numpy(flame_data['flame']['exp']).to(torch.float64)

            transform_mat = np.eye(4)
            transform_mat[:3, :3] = flame_data['opencv']['R'][0]
            transform_mat[:3,  3] = flame_data['opencv']['t'][0]
            transform_mat = inv_cam_extri @ transform_mat # transform under frame0
            global_rot = torch.from_numpy(transform_mat[:3, :3]).unsqueeze(0)
            global_transl = torch.from_numpy(transform_mat[:3,  3]).unsqueeze(0)

            shapes.append(shape)
            exps.append(exp)
            jaw_poses.append(jaw_pose)
            eye_poses.append(eye_pose)
            eyelid_params.append(eyelid_param)
            global_rots.append(global_rot)
            global_transls.append(global_transl)
        
        self.shapes = torch.cat(shapes, dim=0)
        self.exps = torch.cat(exps, dim=0)
        self.eyelid_params = torch.cat(eyelid_params, dim=0)
        self.global_rots = torch.cat(global_rots, dim=0)
        self.global_transls = torch.cat(global_transls, dim=0)

        jaw_poses = torch.cat(jaw_poses, dim=0)
        eye_poses = torch.cat(eye_poses, dim=0)
        self.jaw_poses = rotation_6d_to_matrix(jaw_poses.cuda()).cpu()
        self.eye_poses = rotation_6d_to_matrix(eye_poses.reshape(-1, 2, 6).cuda()).cpu()
        self.neck_poses = torch.eye(3, dtype=torch.float64).unsqueeze(0).repeat(self.exps.shape[0], 1, 1)
        self.pose_transformed = False
        print("Load FLAME params")

    def transform_flame_params(self, flame_model: FLAME):
        shapes = self.shapes.cuda()
        exps = self.exps.cuda()

        joints = flame_model.compute_joint_location(shapes, exps)
        neck_joints = joints[:, 1]
        global_transls = torch.matmul(self.global_rots.cuda(), neck_joints.unsqueeze(-1)).squeeze(-1) - neck_joints
        global_transls += self.global_transls.cuda()

        self.global_transls = global_transls.cpu()
        self.global_rots, self.neck_poses = self.neck_poses, self.global_rots
        self.pose_transformed = True
        print("Transform FLAME params")

    def precompute_bind_mesh(self, flame_model: FLAME):
        verts = flame_model( # blendshape & articulate transform
            shape_params=self.shapes.cuda(),
            expression_params=self.exps.cuda(),
            neck_pose_params=self.neck_poses.cuda(),
            jaw_pose_params=self.jaw_poses.cuda(),
            eye_pose_params=self.eye_poses.cuda(),
            eyelid_params=self.eyelid_params.cuda()
        )
        verts = torch.matmul(verts, self.global_rots.transpose(-1, -2).cuda()) # global transform
        verts += self.global_transls.unsqueeze(1).cuda()
        self.mesh_verts = verts.to(dtype=torch.float32, device='cpu')

    def precompute_blend_weight(self, use_shape_weight: bool, use_pose_weight: bool):
        poses = self.neck_poses if self.pose_transformed else self.global_rots
        # shape_weight = self.exps[:, :50]
        shape_weight = self.exps[:, :100] # extra exp coeffs
        pose_weight = torch.cat([
            poses[:, :2, :].reshape(-1, 6), # matrix to rot6d
            self.jaw_poses[:, :2, :].reshape(-1, 6), # matrix to rot6d
            self.eye_poses[:, :, :2, :].reshape(-1, 12), # matrix to rot6d
            self.eyelid_params,
            self.global_transls
        ], dim=1)
        blend_weights = []
        if use_shape_weight: blend_weights.append(shape_weight)
        if use_pose_weight: blend_weights.append(pose_weight)
        self.blend_weight = torch.cat(blend_weights, dim=1).to(dtype=torch.float32, device='cpu')
