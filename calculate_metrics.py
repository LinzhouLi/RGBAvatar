import os
import sys
import yaml
import torch
import lpips
import numpy as np
from tqdm import tqdm
import nvdiffrast.torch as dr
from argparse import ArgumentParser
from torch.utils.data import DataLoader

from utils import psnr, ssim, Struct
from diff_renderer import render_gs_batch
from model import BindingModel, FLAMEBindingModel, FuHeadBindingModel
from dataset import FLAMEDataset, FuHeadDataset
from submodules.flame import FLAME, FlameConfig
from submodules.fuhead import FuHead
from camera import IntrinsicsCamera, Camera


def compute_metrics(dataset: FLAMEDataset, camera: Camera, gaussian_model: BindingModel):
    bg_color = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device='cuda')
    perceptual_model = lpips.LPIPS(net='vgg').cuda()

    l1_error_vals = torch.zeros([len(dataset)], dtype=torch.float32, device='cuda')
    l2_error_vals = torch.zeros([len(dataset)], dtype=torch.float32, device='cuda')
    psnr_vals = torch.zeros([len(dataset)], dtype=torch.float32, device='cuda')
    ssim_vals = torch.zeros([len(dataset)], dtype=torch.float32, device='cuda')
    lpips_vals = torch.zeros([len(dataset)], dtype=torch.float32, device='cuda')

    i = 0
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False)
    progress_bar = tqdm(range(len(dataset)), desc="Computing Metrics")
    with torch.no_grad():
        for data in dataloader:
            gt_image = data['image'].to(device='cuda', dtype=torch.float32) / 255.0
            gt_image, gt_mask = gt_image[:, :3], gt_image[:, 3:]
            gt_image = gt_image * gt_mask + bg_color.reshape(1, 3, 1, 1) * (1.0 - gt_mask)
            mesh = data['mesh'].cuda()
            blend_weight = data['blend_weight'].cuda()
            bs = gt_image.shape[0]

            gaussian = gaussian_model.gaussian_deform_batch(mesh, blend_weight)
            image = render_gs_batch(camera, bg_color, gaussian)["color"]
            l1_error_vals[i:i+bs] = torch.abs(image - gt_image).reshape(bs, -1).mean(dim=1)
            l2_error_vals[i:i+bs] = torch.square(image - gt_image).reshape(bs, -1).mean(dim=1)
            psnr_vals[i:i+bs] = psnr(image, gt_image).reshape(bs)
            ssim_vals[i:i+bs] = ssim(image, gt_image, size_average=False)
            lpips_vals[i:i+bs] = perceptual_model(image * 2.0 - 1.0, gt_image * 2.0 - 1.0).reshape(bs, -1).mean(dim=1)
            i += bs
            progress_bar.update(bs)

    progress_bar.close()
    l1_error_vals = l1_error_vals.cpu().numpy()
    l2_error_vals = l2_error_vals.cpu().numpy()
    psnr_vals = psnr_vals.cpu().numpy()
    ssim_vals = ssim_vals.cpu().numpy()
    lpips_vals = lpips_vals.cpu().numpy()

    return {
        'l1_error': l1_error_vals,
        'l2_error': l2_error_vals,
        'psnr': psnr_vals,
        'ssim': ssim_vals,
        'lpips': lpips_vals
    }


def compute_training_testing_metrics(metrics, split):
    training_l1_error = np.mean(metrics['l1_error'][:split])
    training_l2_error = np.mean(metrics['l2_error'][:split])
    training_psnr = np.mean(metrics['psnr'][:split])
    training_ssim = np.mean(metrics['ssim'][:split])
    training_lpips = np.mean(metrics['lpips'][:split])
    testing_l1_error = np.mean(metrics['l1_error'][split:])
    testing_l2_error = np.mean(metrics['l2_error'][split:])
    testing_psnr = np.mean(metrics['psnr'][split:])
    testing_ssim = np.mean(metrics['ssim'][split:])
    testing_lpips = np.mean(metrics['lpips'][split:])
    result = "[Training Set]\n" + \
             f"L1 ERROR: {round(training_l1_error, 6):.6f}\n" + \
             f"L2 ERROR: {round(training_l2_error, 7):.7f}\n" + \
             f"PSNR: {round(training_psnr, 2):.2f}\n" + \
             f"SSIM: {round(training_ssim, 4):.4f}\n" + \
             f"LPIPS: {round(training_lpips, 4):.4f}\n" + \
             "[Test Set]\n" + \
             f"L1 ERROR: {round(testing_l1_error, 6):.6f}\n" + \
             f"L2 ERROR: {round(testing_l2_error, 7):.7f}\n" + \
             f"PSNR: {round(testing_psnr, 2):.2f}\n" + \
             f"SSIM: {round(testing_ssim, 4):.4f}\n" + \
             f"LPIPS: {round(testing_lpips, 4):.4f}"
    return result


if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluation script parameters")
    parser.add_argument("--subject", type=str, default="bala")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--work_name", type=str, required=True)
    parser.add_argument("--split", type=int, default=-350)
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

    metrics = compute_metrics(dataset, camera, gaussian_model)
    metric_file = os.path.join(output_path, "metrics.npz")
    np.savez(
        metric_file, 
        l1_error=metrics['l1_error'], 
        l2_error=metrics['l2_error'], 
        psnr=metrics['psnr'], 
        ssim=metrics['ssim'], 
        lpips=metrics['lpips']
    )

    split = (len(dataset) + args.split) % len(dataset)
    result_metrics = compute_training_testing_metrics(metrics, split)
    print(result_metrics)
    with open(os.path.join(output_path, "metrics.txt"), "w") as f:
        f.write(result_metrics)