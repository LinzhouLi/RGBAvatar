import torch
from torch.optim import Adam, AdamW
from torch.utils.tensorboard import SummaryWriter
import lpips

from diff_renderer import BatchGaussianRenderer
from camera import Camera
from utils import Struct, l1_loss, ssim, get_expon_lr_func, create_window, _ssim
from .binding import BindingModel


class Reconstruction:
    def __init__(self,
        camera: Camera,
        gaussian_model: BindingModel,
        batch_size: int,
        recon_config: Struct,
        tb_writer: SummaryWriter = None
    ):
        self.batch_size = batch_size
        self.recon_config = recon_config
        self.tb_writer = tb_writer
        self.camera = camera
        self.gaussian_model = gaussian_model
        self.bg_color = torch.tensor(self.recon_config.bg_color, dtype=torch.float32, device='cuda')
        self.batch_gaussian_renderer = BatchGaussianRenderer(
            bg_color=self.bg_color,
            static_camera=camera, 
            num_gaussians=gaussian_model.num_gaussian,
            batch_size=batch_size
        )
        gs_params, bs_params, adapter_params = gaussian_model.training_params(self.recon_config)
        params = gs_params + bs_params + adapter_params
        self.optimizer = Adam(params=params, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=self.recon_config.position_lr * self.recon_config.scene_extent,
            lr_final=self.recon_config.position_lr_final * self.recon_config.scene_extent,
            lr_delay_mult=self.recon_config.position_lr_delay_mult,
            max_steps=self.recon_config.position_lr_max_steps
        )
        self.iteration = 0
        self.perceptual_model = None
        self.ssim_window = None
        self.global_lr_scale = 1.0


    def perceptual_loss(self, image: torch.Tensor, gt_image: torch.Tensor):
        if self.iteration > 20000: # hard coded
            if self.perceptual_model is None: # load lpips model
                self.perceptual_model = lpips.LPIPS(net='vgg').cuda()
            image = image * 2.0 - 1.0
            gt_image = gt_image * 2.0 - 1.0
            return self.perceptual_model(image, gt_image).mean()
        else:
            return 0.0
    

    def ssim_loss(self, image: torch.Tensor, gt_image: torch.Tensor):
        if self.ssim_window is None: # create ssim window
            self.ssim_window = create_window(11, 3).to(device=image.device, dtype=image.dtype)
        return 1.0 - _ssim(image, gt_image, self.ssim_window, 11, 3, size_average=True)
    

    def update_learning_rate(self):
        xyz_lr = self.xyz_scheduler_args(self.iteration) * self.recon_config.scene_extent
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz": 
                param_group['lr'] = xyz_lr * self.global_lr_scale
            elif param_group["name"] == "opacity": 
                param_group['lr'] = self.recon_config.opacity_lr * self.global_lr_scale
            elif param_group["name"] == "scaling": 
                param_group['lr'] = self.recon_config.scaling_lr * self.global_lr_scale
            elif param_group["name"] == "rotation": 
                param_group['lr'] = self.recon_config.rotation_lr * self.global_lr_scale
            elif param_group["name"] == "f_dc": 
                param_group['lr'] = self.recon_config.feature_lr * self.global_lr_scale
            elif param_group["name"] == "xyz_b": 
                param_group['lr'] = xyz_lr * self.recon_config.position_b_lr_scale * self.global_lr_scale
            elif param_group["name"] == "rotation_b": 
                param_group['lr'] = self.recon_config.rotaion_b_lr_scale * self.recon_config.rotation_lr * self.global_lr_scale
            elif param_group["name"] == "f_dc_b": 
                param_group['lr'] = self.recon_config.feature_b_lr_scale * self.recon_config.feature_lr * self.global_lr_scale
            elif param_group["name"] == "weight_module": 
                param_group['lr'] = self.recon_config.weight_module_lr * self.global_lr_scale

    def step(self, 
        gt_image: torch.Tensor, 
        template_mesh: torch.Tensor,
        blend_weight: torch.Tensor
    ):
        batch_size = gt_image.shape[0]
        self.update_learning_rate()
        gt_rgb, gt_mask = gt_image[:, :3], gt_image[:, 3:4]
        
        # background processing
        if self.recon_config.random_bg_color: bg_color = torch.rand([batch_size, 3, 1, 1], dtype=torch.float32, device='cuda')
        else: bg_color = self.bg_color.reshape(1, 3, 1, 1).repeat(batch_size, 1, 1, 1)
        gt_rgb = gt_rgb * gt_mask + bg_color * (1.0 - gt_mask)

        # blend & bind
        blend_weight = None if self.iteration < self.recon_config.blend_start_iter else blend_weight
        gaussian = self.gaussian_model.gaussian_deform_batch(template_mesh, blend_weight)
        self.optimizer.zero_grad(set_to_none = True)

        # batch render
        render_pkg = self.batch_gaussian_renderer.render(bg_color, gaussian, gt_image)
        image, alpha = render_pkg["color"], render_pkg["alpha"]

        # loss
        l1_loss_val = l1_loss(image, gt_rgb) if self.recon_config.lambda_l1 > 0.0 else 0.0
        ssim_loss_val = self.ssim_loss(image, gt_rgb) if self.recon_config.lambda_ssim > 0.0 else 0.0
        lpips_loss_val = self.perceptual_loss(image, gt_rgb) if self.recon_config.lambda_lpips > 0.0 else 0.0
        alpha_loss_val = l1_loss(alpha, gt_mask) if self.recon_config.lambda_alpha > 0.0 else 0.0
        sparsity_loss_val = self.gaussian_model.sparsity_loss(blend_weight) if self.recon_config.lambda_sparsity > 0.0 else 0.0
        orth_loss_val = self.gaussian_model.orth_loss() if self.recon_config.lambda_orth > 0.0 and blend_weight is not None else 0.0
        total_loss = self.recon_config.lambda_l1 * l1_loss_val + \
            self.recon_config.lambda_ssim * ssim_loss_val + \
            self.recon_config.lambda_lpips * lpips_loss_val + \
            self.recon_config.lambda_alpha * alpha_loss_val + \
            self.recon_config.lambda_sparsity * sparsity_loss_val +\
            self.recon_config.lambda_orth * orth_loss_val
        
        # optimize
        total_loss.backward()
        self.optimizer.step()

        # fast forward
        if self.recon_config.use_fast_forward:
            self.gaussian_model.fast_forward(render_pkg["est_color"], render_pkg["est_weight"])

        # log
        if self.tb_writer is not None:
            if self.recon_config.lambda_l1 > 0.0:
                self.tb_writer.add_scalar('train_loss/l1_loss', l1_loss_val.item(), self.iteration)
            if self.recon_config.lambda_ssim > 0.0:
                self.tb_writer.add_scalar('train_loss/ssim_loss', ssim_loss_val.item(), self.iteration)
            if self.recon_config.lambda_lpips > 0.0:
                self.tb_writer.add_scalar('train_loss/lpips_loss', lpips_loss_val.item(), self.iteration)
            if self.recon_config.lambda_alpha > 0.0:
                self.tb_writer.add_scalar('train_loss/alpha_loss', alpha_loss_val.item(), self.iteration)
            if self.recon_config.lambda_alpha > 0.0:
                self.tb_writer.add_scalar('train_loss/orth_loss', orth_loss_val.item(), self.iteration)
            self.tb_writer.add_scalar('train_loss/total_loss', total_loss.item(), self.iteration)
            
        self.iteration += batch_size