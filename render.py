import os
import sys
import yaml
import torch
from tqdm import tqdm
import nvdiffrast.torch as dr
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

from diff_renderer import render_gs_batch
from model import BindingModel, FLAMEBindingModel, FuHeadBindingModel
from dataset import FLAMEDataset, FuHeadDataset
from submodules.flame import FLAME, FlameConfig
from submodules.fuhead import FuHead
from camera import IntrinsicsCamera, Camera
from utils import Struct


def save_image(image_data, output_img_path, index):
    Image.fromarray(image_data).save(os.path.join(output_img_path, "{:0>5d}.png".format(index)))


def render(gaussian_model: BindingModel, dataset: FLAMEDataset | FuHeadDataset, camera: Camera, bg_color: torch.Tensor, alpha: bool, output_path):
    output_img_path = os.path.join(output_path, "render_image")
    os.makedirs(output_img_path, exist_ok=True)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False)

    i = 0
    progress_bar = tqdm(range(len(dataset)), desc="Rendering")
    with ThreadPoolExecutor() as executor:
        for data in dataloader:
            mesh = data['mesh'].cuda()
            blend_weight = data['blend_weight'].cuda()
            bs = mesh.shape[0]
            gaussian = gaussian_model.gaussian_deform_batch(mesh, blend_weight)
            render_pkg = render_gs_batch(camera, bg_color, gaussian)
            if alpha: image = torch.cat([render_pkg["color"], render_pkg["alpha"]], dim=1)
            else: image = render_pkg["color"]

            image = (image.permute(0, 2, 3, 1) * 255.0).to(dtype=torch.uint8, device='cpu').numpy()
            for j in range(bs):
                executor.submit(save_image, image[j], output_img_path, i)
                i += 1
            progress_bar.update(bs)
    progress_bar.close()
    print("Done!")


if __name__ == "__main__":
    parser = ArgumentParser(description="Rendering script parameters")
    parser.add_argument("--subject", type=str, default="bala")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--work_name", type=str, required=True)
    parser.add_argument("--white_bg", action="store_true")
    parser.add_argument("--alpha", action="store_true")
    args = parser.parse_args(sys.argv[1:])

    glctx = dr.RasterizeGLContext()
    output_path = os.path.join(args.output_dir, args.subject, args.work_name)
    with open(os.path.join(output_path, "config.yaml")) as f: 
        config = yaml.load(f, Loader=yaml.FullLoader)
    data_path = os.path.join(config['data_dir'], args.subject)

    if config['template_type'] == 'Flame':
        flame_model = FLAME(FlameConfig()).cuda()
        dataset = FLAMEDataset(flame_model, data_path, split='all', **config['dataset'])
        gaussian_model = FLAMEBindingModel(Struct(**config['model']), flame_model, glctx)
    elif config['template_type'] == 'FuHead':
        fuhead_model = FuHead().cuda()
        dataset = FuHeadDataset(fuhead_model, data_path, split='all', **config['dataset'])
        gaussian_model = FuHeadBindingModel(Struct(**config['model']), fuhead_model, glctx)
    else:
        raise NotImplementedError

    gaussian_model.load_ply(os.path.join(output_path, "model.ply"))
    camera = IntrinsicsCamera(
        K=dataset.camera_intri, 
        R=dataset.camera_extri[:3, :3], 
        T=dataset.camera_extri[:3,  3],
        width=dataset.image_width, 
        height=dataset.image_height
    ).cuda()
    if args.white_bg:
        bg_color = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device='cuda')
    else:
        bg_color = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device='cuda')

    render(gaussian_model, dataset, camera, bg_color, args.alpha, output_path)