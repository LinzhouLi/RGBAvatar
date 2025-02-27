from dataclasses import dataclass
import torch

@dataclass
class FlameConfig:
    flame_geom_path = "./data/FLAME2020/generic_model.pkl" # flame model path
    flame_uv_path = "./data/FLAME2020/flame_uv.npz" # flame uv data path
    flame_mask_path = "./data/FLAME2020/FLAME_masks.pkl"
    flame_lmk_path = "./data/FLAME2020/landmark_embedding.npy" # Landmark embeddings path for FLAME
    mediapipe_lmk_path = "./data/FLAME2020/mediapipe_landmark_embedding.npz"
    l_eyelid_path = "./data/FLAME2020/l_eyelid.npy"
    r_eyelid_path = "./data/FLAME2020/r_eyelid.npy"
    num_shape_params = 300 # the number of shape parameters
    num_exp_params = 100 # the number of expression parameters
    batch_size = 1 # Training batch size.
    dtype = torch.float64
    add_mouth = True
