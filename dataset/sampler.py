import numpy as np
import torch


class Sampler:
    def __init__(self,
        local_pool_max_size = 150,
        global_pool_max_size = 1000,
        max_global_ratio = 0.8,
        local_sample_coef = 0.5,
        force_opt_current = False  ,
        force_local = False,
        force_global = False
    ):
        self.local_pool_max_size = local_pool_max_size
        self.global_pool_max_size = global_pool_max_size
        self.max_global_ratio = max_global_ratio
        self.local_sample_coef = local_sample_coef
        self.force_opt_current = force_opt_current
        self.force_local = force_local
        self.force_global = force_global

        self.local_sampling_weights = np.exp(self.local_sample_coef * np.arange(self.local_pool_max_size) / self.local_pool_max_size)
        self.local_pool = np.zeros(self.local_pool_max_size, dtype=np.int32)
        self.global_pool = np.zeros(self.global_pool_max_size, dtype=np.int32)
        self.global_data_count = 0
        self.local_pool_ptr = 0
        self.local_pool_size = 0
        self.sample_count = 0

        local_weight_sum = self.local_sampling_weights.sum()
        global_weight_sum = local_weight_sum * self.max_global_ratio / (1.0 - self.max_global_ratio)
        global_sample_weight = self.local_sampling_weights[0]
        self.max_ratio_global_sample_count = max(int(global_weight_sum / global_sample_weight), 1)
        print("Max Global Sample Ratio:", self.max_global_ratio)
        print("Max Ratio Global Sample Count:", self.max_ratio_global_sample_count)
    
    @property
    def global_pool_size(self):
        return min(self.global_data_count, self.global_pool_max_size)
    
    def _update_global_pool(self, data: int): # Reservoir Sampling
        if self.global_data_count < self.global_pool_max_size:
            self.global_pool[self.global_data_count] = data
        else:
            i = np.random.randint(0, self.global_data_count)
            if i < self.global_pool_max_size:
                self.global_pool[i] = data
        self.global_data_count += 1

    def add_data(self, data: int):
        self.sample_count += 1
        if self.force_global:
            self._update_global_pool(data)
            return
        
        if self.local_pool_size == self.local_pool_max_size:
            if not self.force_local:
                oldest_local_data = self.local_pool[self.local_pool_ptr]
                self._update_global_pool(oldest_local_data)
            self.local_pool[self.local_pool_ptr] = data
            self.local_pool_ptr = (self.local_pool_ptr + 1) % self.local_pool_max_size
        else:
            self.local_pool[self.local_pool_ptr] = data
            self.local_pool_ptr = (self.local_pool_ptr + 1) % self.local_pool_max_size
            self.local_pool_size += 1

    def sample_local(self, num_samples: int):
        if self.force_opt_current:
            assert num_samples >= 1
            if num_samples == 1: return [self.local_pool_ptr - 1] # return current sample
        
        if self.local_pool_ptr == 0:
            local_sampling_weights = self.local_sampling_weights.copy()
        else:
            local_sampling_weights = np.concatenate([
                self.local_sampling_weights[-self.local_pool_ptr:],
                self.local_sampling_weights[:-self.local_pool_ptr]
            ])

        if self.force_opt_current:
            local_sampling_weights[self.local_pool_ptr - 1] = 0.0 # exclude current sample
            local_sampling_weights = local_sampling_weights[:self.local_pool_size]
            local_sampling_weights /= local_sampling_weights.sum()
            local_pool_indices = np.random.choice(range(self.local_pool_size), num_samples - 1, replace=False, p=local_sampling_weights)
            
            local_pool_indices = local_pool_indices.tolist()
            local_pool_indices.append(self.local_pool_ptr - 1) # add current sample with 100% probability
        else:
            local_sampling_weights = local_sampling_weights[:self.local_pool_size]
            local_sampling_weights /= local_sampling_weights.sum()
            local_pool_indices = np.random.choice(range(self.local_pool_size), num_samples, replace=False, p=local_sampling_weights).tolist()
        
        return local_pool_indices
    
    def sample_global(self, num_samples: int):
        assert self.global_pool_size >= num_samples
        global_pool_indices = np.random.choice(range(self.global_pool_size), num_samples, replace=False)
        return global_pool_indices.tolist()
        
    def split_batch(self, batch_size: int):
        if self.force_local: return batch_size, 0
        if self.force_global: return 0, batch_size

        assert self.local_pool_size >= batch_size
        global_sample_ratio = self.max_global_ratio * min((self.sample_count - self.local_pool_size) / self.max_ratio_global_sample_count, 1.0)
        num_global_samples = 0
        for _ in range(batch_size):
            if np.random.rand() < global_sample_ratio: num_global_samples += 1
        if self.force_opt_current:
            num_global_samples = min(num_global_samples, batch_size - 1) # leave at leaset one sample for current frame
        num_global_samples = min(num_global_samples, self.global_pool_size)
        num_local_samples = batch_size - num_global_samples
        return num_local_samples, num_global_samples

    def sample(self, batch_size: int):
        num_local_samples, num_global_samples = self.split_batch(batch_size)
        samples = []

        # sampling from local pool
        if num_local_samples > 0:
            local_pool_indices = self.sample_local(num_local_samples)
            for local_pool_index in local_pool_indices: 
                samples.append(self.local_pool[local_pool_index])

        # sampling from global pool
        if num_global_samples > 0:
            global_pool_indices = self.sample_global(num_global_samples)
            for global_pool_index in global_pool_indices: 
                samples.append(self.global_pool[global_pool_index])
        return samples


class ErrorSampler(Sampler):
    def __init__(self, 
        num_frames: int,
        error_sample_strength: float,
        local_pool_max_size = 150,
        global_pool_max_size = 1000,
        max_global_ratio = 0.8,
        local_sample_coef = 0.5,
        force_opt_current = False
    ):
        super().__init__(local_pool_max_size, global_pool_max_size, max_global_ratio, local_sample_coef, force_opt_current)
        self.frame_errors = np.zeros(num_frames, dtype=np.float32)
        self.error_sample_coef = error_sample_strength

    def sample_global(self, num_samples: int):
        assert self.global_pool_size >= num_samples
        frame_errors = self.frame_errors[self.global_pool[:self.global_pool_size]]
        global_sampling_weights = np.exp(self.error_sample_coef * frame_errors)
        global_sampling_weights /= global_sampling_weights.sum()
        global_pool_indices = np.random.choice(range(self.global_pool_size), num_samples, replace=False, p=global_sampling_weights)
        return global_pool_indices.tolist()

    def sample(self, batch_size: int):
        num_local_samples, num_global_samples = self.split_batch(batch_size)
        samples = []

        # sampling from local pool
        if num_local_samples > 0:
            local_pool_indices = self.sample_local(num_local_samples)
            for local_pool_index in local_pool_indices: 
                samples.append(self.local_pool[local_pool_index])

        # sampling from global pool
        if num_global_samples > 0:
            global_pool_indices = self.sample_global(num_global_samples)
            for global_pool_index in global_pool_indices: 
                samples.append(self.global_pool[global_pool_index])
        return samples


class CoeffSampler(Sampler):
    def __init__(self, blend_weights: torch.Tensor, proj_module: torch.nn.Module):
        super().__init__()
        self.blend_weights = blend_weights.cpu()
        self.proj_module = proj_module
        self.coef_sample_coef = 0.0001

    def sample_global(self, num_samples: int, local_blend_weights: torch.Tensor):
        assert self.global_pool_size >= num_samples
        global_blend_weights = self.blend_weights[self.global_pool[:self.global_pool_size]]
        global_coeff = self.proj_module(global_blend_weights.cuda())
        local_coeff = self.proj_module(local_blend_weights.cuda())
        coeff_similarity = torch.abs(global_coeff.unsqueeze(1) * local_coeff.unsqueeze(0)).mean([1,2])
        coeff_similarity = coeff_similarity.cpu().numpy()

        global_sampling_weights = np.exp(self.coef_sample_coef * coeff_similarity)
        global_sampling_weights /= global_sampling_weights.sum()
        global_pool_indices = np.random.choice(range(self.global_pool_size), num_samples, replace=False, p=global_sampling_weights)
        return global_pool_indices.tolist()

    @torch.no_grad()
    def sample(self, batch_size: int):
        num_local_samples, num_global_samples = self.split_batch(batch_size)
        samples = []

        # sampling from local pool
        if num_local_samples > 0:
            local_pool_indices = self.sample_local(num_local_samples)
            for local_pool_index in local_pool_indices: 
                samples.append(self.local_pool[local_pool_index])

        # sampling from global pool
        if num_global_samples > 0:
            local_blend_weights = self.blend_weights[samples]
            global_pool_indices = self.sample_global(num_global_samples, local_blend_weights)
            for global_pool_index in global_pool_indices: 
                samples.append(self.global_pool[global_pool_index])
        return samples