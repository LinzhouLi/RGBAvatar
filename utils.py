import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

    def __str__(self):
        info = ["{}={}".format(k, v) for k, v in self.__dict__.items()]
        return "Struct({})".format(", ".join(info))

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)
    window = window.to(device=img1.device, dtype=img1.dtype)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    
def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)

def quaternion_multiply(p: torch.Tensor, q: torch.Tensor):
    """
    Returns the product of two quaternions.
    Adapted from roma.

    Args:
        p, q (...x4 tensor, WXYZ convention): batch of quaternions.
    Returns:
        batch of quaternions (...x4 tensor, WXYZ convention).
    """
    vector = (
        p[..., None, 0] * q[..., 1:] + 
        q[..., None, 0] * p[..., 1:] +
        torch.cross(p[..., 1:], q[..., 1:], dim=-1)
    )
    last = p[..., 0] * q[..., 0] - torch.sum(p[..., 1:] * q[..., 1:], axis=-1)
    return torch.cat((last[..., None], vector), dim=-1)

def matrix_to_quaternion(R: torch.Tensor):
    """
    Converts rotation matrix to unit quaternion representation.
    Adapted from roma.

    Args:
        R (...x3x3 tensor): batch of rotation matrices.
    Returns:
        batch of unit quaternions (...x4 tensor, WXYZ convention).
    """
    batch_shape = R.shape[:-2]
    matrix = R.flatten(end_dim=-3) if len(batch_shape) > 0 else R.unsqueeze(0)
    num_rotations, D1, D2 = matrix.shape
    assert((D1, D2) == (3,3)), "Input should be a Bx3x3 tensor."

    decision_matrix = torch.empty((num_rotations, 4), dtype=matrix.dtype, device=matrix.device)
    decision_matrix[:, :3] = matrix.diagonal(dim1=1, dim2=2)
    decision_matrix[:, -1] = decision_matrix[:, :3].sum(axis=1)
    choices = decision_matrix.argmax(axis=1)

    ind1 = torch.nonzero(choices != 3, as_tuple=True)[0]
    ind2 = torch.nonzero(choices == 3, as_tuple=True)[0]
    quat = torch.empty((num_rotations, 4), dtype=matrix.dtype, device=matrix.device)

    i = choices[ind1]
    j = (i + 1) % 3
    k = (j + 1) % 3

    quat[ind1, i + 1] = 1 - decision_matrix[ind1, -1] + 2 * matrix[ind1, i, i]
    quat[ind1, j + 1] = matrix[ind1, j, i] + matrix[ind1, i, j]
    quat[ind1, k + 1] = matrix[ind1, k, i] + matrix[ind1, i, k]
    quat[ind1, 0] = matrix[ind1, k, j] - matrix[ind1, j, k]

    quat[ind2, 1] = matrix[ind2, 2, 1] - matrix[ind2, 1, 2]
    quat[ind2, 2] = matrix[ind2, 0, 2] - matrix[ind2, 2, 0]
    quat[ind2, 3] = matrix[ind2, 1, 0] - matrix[ind2, 0, 1]
    quat[ind2, 0] = 1 + decision_matrix[ind2, -1]

    quat = F.normalize(quat, dim=1)
    quat = quat.reshape(batch_shape + quat.shape[1:]) if len(batch_shape) > 0 else quat.squeeze(0)
    return quat

# @torch.compile # Found NVIDIA GeForce GTX 1080 Ti which is too old to be supported by the triton GPU compiler
def compute_face_tbn(face_verts: torch.Tensor, face_uvs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    v0, v1, v2 = face_verts.unbind(-2)
    uv0, uv1, uv2 = face_uvs.unbind(-2)

    edge1 = v1 - v0
    edge2 = v2 - v0

    deltaUV1 = uv1 - uv0
    deltaUV2 = uv2 - uv0

    normal = torch.cross(edge1, edge2, dim=-1)
    area = normal.norm(dim=-1, keepdim=True) * 0.5

    f = 1.0 / (deltaUV1[..., 0] * deltaUV2[..., 1] - deltaUV2[..., 0] * deltaUV1[..., 1])
    f = f.unsqueeze(1)

    tangent = f * (deltaUV2[..., 1].unsqueeze(1) * edge1 - deltaUV1[..., 1].unsqueeze(1) * edge2)
    bitangent = f * (-deltaUV2[..., 0].unsqueeze(1) * edge1 + deltaUV1[..., 0].unsqueeze(1) * edge2)

    tbn = torch.stack([tangent, bitangent, normal], dim=-1) # [F, 3, 3]
    tbn = F.normalize(tbn, dim=-2) # normalize tbn in one step
    return tbn

def gather_vert_attributes(
    face_attrs: torch.Tensor, # [F, C]
    face_weights: torch.Tensor, # [F, 1]
    faces: torch.Tensor # [F, 3]
) -> torch.Tensor:
    num_verts = faces.max()
    vert_attrs = torch.zeros([num_verts + 1, face_attrs.shape[1]], dtype=face_attrs.dtype, device=face_attrs.device)
    weighted_face_attrs = face_attrs * face_weights

    vert_attrs = vert_attrs.index_add(0, faces[:, 0], weighted_face_attrs)
    vert_attrs = vert_attrs.index_add(0, faces[:, 1], weighted_face_attrs)
    vert_attrs = vert_attrs.index_add(0, faces[:, 2], weighted_face_attrs)
    return vert_attrs

def rgb2sh0(rgb: torch.Tensor) -> torch.Tensor:
    return (rgb - 0.5) / 0.28209479177387814


def flatten_model_params(model: torch.nn.Module):
    flat_params = []
    for param in model.state_dict().values():
        flat_params.append(param.view(-1))
    return torch.cat(flat_params)

def load_flattened_model_params(flat_params: torch.Tensor, model: torch.nn.Module):
    state_dict = model.state_dict()
    offset = 0
    for key, param in state_dict.items():
        numel = param.numel()
        new_param = flat_params[offset:offset+numel].view(param.size())
        state_dict[key].copy_(new_param)
        offset += numel
    model.load_state_dict(state_dict)

def model_size(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters())


def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x / (1 - x))

def indentity(x: torch.Tensor) -> torch.Tensor:
    return x


def smooth(x: np.ndarray, weight = 0.9):
    last = x[0]
    smoothed = []
    for point in x:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)


def average_rotation(R: np.ndarray):
    R_avg = np.mean(R, axis=0)
    U, _, Vt = np.linalg.svd(R_avg)
    R_avg_corrected = U @ Vt
    return R_avg_corrected
