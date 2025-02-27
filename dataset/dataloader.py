import random
import torch
import numpy as np
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Sampler as TorchSampler


class ContinuousRandomSampler(TorchSampler):
    def __init__(self, dataset):
        self.dataset = dataset
        self.data_size = len(dataset)

    def __iter__(self):
        while True:
            yield random.randint(0, self.data_size - 1)


class DataLoader:
    def __init__(self, dataset, batch_size: int = 10):
        self.dataset = dataset
        self.batch_size = batch_size

        if self.dataset.preload:
            self.stream = torch.cuda.Stream()
            self.data = None
            self.prefetch()
        else:
            sampler = ContinuousRandomSampler(dataset)
            dataloader = TorchDataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=8)
            self.dataloader_iter = iter(dataloader)


    def prefetch(self):
        with torch.cuda.stream(self.stream):
            data_indices = np.random.choice(range(len(self.dataset)), self.batch_size, replace=False).tolist()
            mesh = self.dataset.mesh_verts[data_indices].to(device='cuda', non_blocking=True)
            blend_weight = self.dataset.blend_weight[data_indices].to(device='cuda', non_blocking=True)
            images = [ # faster than cpu data indexing
                self.dataset.images[i].to(device='cuda', non_blocking=True)
                for i in data_indices
            ]
            image = torch.stack(images).contiguous().to(dtype=torch.float32) / 255.0
            self.data = { 'image': image, 'mesh': mesh, 'blend_weight': blend_weight }


    def get_batch(self):
        if self.dataset.preload:
            assert self.data is not None
            torch.cuda.current_stream().wait_stream(self.stream)
            data = self.data
            self.prefetch()
            return data['image'], data['mesh'], data['blend_weight']
        else:
            data = next(self.dataloader_iter)
            image = data['image'].to(device='cuda', dtype=torch.float32, non_blocking=True) / 255.0
            mesh = data['mesh'].to(device='cuda', non_blocking=True)
            blend_weight = data['blend_weight'].to(device='cuda', non_blocking=True)
            return image, mesh, blend_weight
