import os
import sys
import time
import yaml
import torch
import shutil
import pygame
import threading
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
import nvdiffrast.torch as dr
from torch.utils.tensorboard import SummaryWriter

from utils import Struct
from camera import IntrinsicsCamera, Camera
from model import FLAMEBindingModel, BindingModel, Reconstruction
from dataset import FLAMEDataset, DataLoader
from dataset.sampler import Sampler
from submodules.flame import FLAME, FlameConfig


def post_process(recon: Reconstruction, dataset: FLAMEDataset, iteration: int):
    batch_size = recon.batch_size
    dataloader = DataLoader(dataset, batch_size=batch_size)

    progress_bar = tqdm(range(iteration), desc="Post Training")
    for i in range((iteration + batch_size - 1) // batch_size):
        image, mesh, blend_weight = dataloader.get_batch()
        recon.step(image, mesh, blend_weight)
        progress_bar.update(batch_size)
    progress_bar.close()


class ReconThread(threading.Thread):
    def __init__(self, 
        dataset: FLAMEDataset, 
        recon_model: Reconstruction,
        sampler: Sampler
    ):
        super().__init__()
        self.recon = recon_model
        self.dataset = dataset
        self.sampler = sampler

        self.start_time = None
        self.stop_time = None
        self.pause_start_time = None
        self.pause_time = 0
        self._stop_flag = False
        self._paused = True
        self._recon_event = threading.Event()
        self._recon_event.clear() # pause thread when initialization

        self.current_frame_id = 0
        self.sample_record = []
        self.step2frame_record = []

        self.data_stream = torch.cuda.Stream()
        self.data = None


    def prefetch(self, data_indices):
        with torch.cuda.stream(self.data_stream):
            mesh = self.dataset.mesh_verts[data_indices].to(device='cuda', non_blocking=True)
            blend_weight = self.dataset.blend_weight[data_indices].to(device='cuda', non_blocking=True)
            images = [ # faster than cpu data indexing
                self.dataset.images[i].to(device='cuda', non_blocking=True)
                for i in data_indices
            ]
            image = torch.stack(images).contiguous().to(dtype=torch.float32) / 255.0
            self.data = { 'image': image, 'mesh': mesh, 'blend_weight': blend_weight, 'data_indices': data_indices }


    def get_batch(self, data_indices):
        assert self.data is not None
        torch.cuda.current_stream().wait_stream(self.data_stream)
        data = self.data
        self.prefetch(data_indices)
        return data['image'], data['mesh'], data['blend_weight'], data['data_indices']

    @property
    def iteration(self):
        return self.recon.iteration

    @property
    def run_time(self):
        if self.start_time is None: return 0 # not started
        elif self.stop_time is None: 
            if self._paused: return self.pause_start_time - self.start_time - self.pause_time # paused
            else: return time.time() - self.start_time - self.pause_time # running
        else: return self.stop_time - self.start_time - self.pause_time # has stopped

    def is_running(self):
        return self.is_alive() and not self._paused

    def stop(self):
        self.stop_time = time.time()
        self._stop_flag = True
        self._recon_event.set()

    def pause(self):
        if not self._paused and self._recon_event.is_set():
            self._recon_event.clear()
            self.pause_start_time = time.time()
            self._paused = True
            print("pause recon thread")

    def resume(self):
        if self._paused and not self._recon_event.is_set():
            self.pause_time += time.time() - self.pause_start_time
            self._recon_event.set()
            self._paused = False
            print("resume recon thread")

    @torch.no_grad()
    def add_frame(self, index):
        self.sampler.add_data(index)
        self.current_frame_id = index

        if index + 1 == self.recon.batch_size: # start recon when frames count reaches batch size
            data_indices = self.sampler.sample(self.recon.batch_size)
            self.prefetch(data_indices)
            self._recon_event.set()
            # print("start recon thread")

    def run(self):
        self._paused = False
        self.start_time = time.time()
        while not self._stop_flag:
            self._recon_event.wait()
            current_frame_id = self.current_frame_id

            prefetch_data_indices = self.sampler.sample(self.recon.batch_size)
            image, mesh, blend_weight, data_indices = self.get_batch(prefetch_data_indices)
            self.recon.step(image, mesh, blend_weight)

            self.sample_record.append(data_indices)
            self.step2frame_record.append(current_frame_id) # not exact


def train_online(
    dataset: FLAMEDataset, gaussian_model: BindingModel, camera: Camera, 
    video_fps: int, output_path: str, train_config, log: bool = False
):
    batch_size = train_config['batch_size']
    tb_writer = SummaryWriter(output_path) if log else None
    recon = Reconstruction(
        camera, gaussian_model, batch_size=batch_size, 
        recon_config=Struct(**train_config['recon']), tb_writer=tb_writer
    )
    sampler = Sampler(**train_config['sampler'])
    recon_thread = ReconThread(dataset, recon, sampler)
    clock = pygame.time.Clock()

    recon_thread.start()
    progress_bar = tqdm(range(len(dataset)), desc="Training")
    for frame_index in range(len(dataset)):
        clock.tick(video_fps)
        recon_thread.add_frame(frame_index)
        progress_bar.update(1)
        progress_bar.set_postfix({
            "Iteration": recon_thread.iteration,
            "Training FPS": recon_thread.iteration / (recon_thread.run_time + 0.0001)
        })

    # stop online training
    recon_thread.stop()
    recon_thread.join()
    progress_bar.close()
    if log: tb_writer.close()

    # record training speed
    result = f"Iteraion: {recon_thread.iteration}\nTraning FPS: {recon_thread.iteration / recon_thread.run_time:.2f}"
    print(result)
    with open(os.path.join(output_path, "speed.txt"), "w") as f:
        f.write(result)

    # post process
    if train_config['post_process']:
        post_process(recon, dataset, train_config['post_process_iteration'])

    # save model
    gaussian_model.save_ply(os.path.join(output_path, "model.ply"))

    np.savetxt(os.path.join(output_path, "sample.txt"), np.array(recon_thread.sample_record), fmt='%d')
    np.savetxt(os.path.join(output_path, "step2frame.txt"), np.array(recon_thread.step2frame_record), fmt='%d')


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--subject", type=str, default="bala")
    parser.add_argument("--work_name", type=str, default=None)
    parser.add_argument("--video_fps", type=int, default=25)
    parser.add_argument("--config", type=str, default="config/online.yaml")
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
    print("Video FPS:", args.video_fps)

    flame_model = FLAME(FlameConfig()).cuda()
    train_dataset = FLAMEDataset(flame_model, data_path, split='train', preload=True, **config['dataset'])
    gaussian_model = FLAMEBindingModel(Struct(**config['model']), flame_model, glctx)
    gaussian_model.initialize()
    camera = IntrinsicsCamera(
        K=train_dataset.camera_intri, 
        R=train_dataset.camera_extri[:3, :3], 
        T=train_dataset.camera_extri[:3,  3],
        width=train_dataset.image_width, 
        height=train_dataset.image_height
    ).cuda()

    train_online(train_dataset, gaussian_model, camera, args.video_fps, output_path, config['train'], args.log)