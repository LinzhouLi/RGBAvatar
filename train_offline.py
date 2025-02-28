import os
import sys
import time
import yaml
import shutil
from tqdm import tqdm
from argparse import ArgumentParser
import nvdiffrast.torch as dr
from torch.utils.tensorboard import SummaryWriter

from utils import Struct
from camera import IntrinsicsCamera, Camera
from model import FLAMEBindingModel, FuHeadBindingModel, Reconstruction, BindingModel
from dataset import FLAMEDataset, FuHeadDataset, DataLoader
from submodules.flame import FLAME, FlameConfig
from submodules.fuhead import FuHead


def train_offline(
    dataset: FLAMEDataset | FuHeadDataset, gaussian_model: BindingModel, camera: Camera, 
    output_path: str, train_config, log: bool = False
):
    batch_size = train_config['batch_size']
    iteration = train_config['iteration']
    tb_writer = SummaryWriter(output_path) if log else None
    dataloader = DataLoader(dataset, batch_size=batch_size)
    recon = Reconstruction(
        camera, gaussian_model, batch_size=batch_size, 
        recon_config=Struct(**train_config['recon']), tb_writer=tb_writer
    )

    print("Iteraion:", iteration)
    print("Dataset size:", len(dataset))
    progress_bar = tqdm(range(iteration), desc="Training")
    for i in range((iteration + batch_size - 1) // batch_size):
        image, mesh, blend_weight = dataloader.get_batch()
        recon.step(image, mesh, blend_weight)
        progress_bar.update(batch_size)

    training_fps = iteration / progress_bar.format_dict['elapsed']
    if log: tb_writer.close()
    progress_bar.close()
    gaussian_model.prune()
    gaussian_model.save_ply(os.path.join(output_path, "model.ply"))

    result = f"Traning FPS: {training_fps:.2f}"
    print(result)
    with open(os.path.join(output_path, "speed.txt"), "w") as f:
        f.write(result)


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--subject", type=str, default="bala")
    parser.add_argument("--work_name", type=str, default=None)
    parser.add_argument("--config", type=str, default="config/offline.yaml")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--preload", action="store_true")
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    with open(args.config) as f: config = yaml.load(f, Loader=yaml.FullLoader)

    glctx = dr.RasterizeGLContext()
    data_path = os.path.join(config['data_dir'], args.subject)
    if args.work_name is None:
        args.work_name = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(int(time.time())))
    output_path = os.path.join(config['output_dir'], args.subject, args.work_name)
    if os.path.exists(output_path): shutil.rmtree(output_path)
    os.makedirs(output_path)
    shutil.copy(args.config, os.path.join(output_path, "config.yaml"))
    print("Output:", output_path)

    if config['template_type'] == 'Flame':
        flame_model = FLAME(FlameConfig()).cuda()
        train_dataset = FLAMEDataset(flame_model, data_path, split=args.split, preload=args.preload, **config['dataset'])
        gaussian_model = FLAMEBindingModel(Struct(**config['model']), flame_model, glctx)
    elif config['template_type'] == 'FuHead':
        fuhead_model = FuHead().cuda()
        train_dataset = FuHeadDataset(fuhead_model, data_path, split=args.split, preload=args.preload, **config['dataset'])
        gaussian_model = FuHeadBindingModel(Struct(**config['model']), fuhead_model, glctx)
    else:
        raise NotImplementedError

    gaussian_model.initialize()
    camera = IntrinsicsCamera(
        K=train_dataset.camera_intri, 
        R=train_dataset.camera_extri[:3, :3], 
        T=train_dataset.camera_extri[:3,  3],
        width=train_dataset.image_width, 
        height=train_dataset.image_height
    ).cuda()

    train_offline(train_dataset, gaussian_model, camera, output_path, config['train'], args.log)