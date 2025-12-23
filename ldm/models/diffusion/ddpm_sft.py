import os
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange, repeat
from contextlib import contextmanager
from functools import partial
from tqdm import tqdm
from torchvision.utils import make_grid
from pytorch_lightning.utilities.distributed import rank_zero_only
from ldm.util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config
from ldm.modules.ema import LitEma
from ldm.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution
from ldm.models.autoencoder import VQModelInterface, IdentityFirstStage, AutoencoderKL
from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from ldm.models.diffusion.ddim import DDIMSampler
from torchvision.transforms import Resize
import torchvision.transforms.functional as TF  
import torch.nn.functional as F
import random
from torch.autograd import Variable
from src.Face_models.encoders.model_irse import Backbone
import dlib
from eval_tool.lpips.lpips import LPIPS
import wandb
from PIL import Image
__conditioning_keys__ = {'concat': 'c_concat',
                         'crossattn': 'c_crossattn',
                         'adm': 'y'}
def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def un_norm_clip(x1):
    x = x1*1.0 # to avoid changing the original tensor or clone() can be used
    reduce=False
    if len(x.shape)==3:
        x = x.unsqueeze(0)
        reduce=True
    x[:,0,:,:] = x[:,0,:,:] * 0.26862954 + 0.48145466
    x[:,1,:,:] = x[:,1,:,:] * 0.26130258 + 0.4578275
    x[:,2,:,:] = x[:,2,:,:] * 0.27577711 + 0.40821073
    
    if reduce:
        x = x.squeeze(0)
    return x

def un_norm(x):
    return (x+1.0)/2.0
 
def save_clip_img(img, path,clip=True):
    if clip:
        img=un_norm_clip(img)
    else:
        img=torch.clamp(un_norm(img), min=0.0, max=1.0)
    img = img.cpu().numpy().transpose((1, 2, 0))
    img = (img * 255).astype(np.uint8)
    img = Image.fromarray(img)
    img.save(path)
    # if clip:
    #     img=TF.normalize(img, mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    # else:  
    #     img=TF.normalize(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])



class IDLoss(nn.Module):
    def __init__(self,opts,multiscale=False):
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace')
        self.opts = opts 
        self.multiscale = multiscale
        self.face_pool_1 = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        # self.facenet=iresnet100(pretrained=False, fp16=False) # changed by sanoojan
        
        self.facenet.load_state_dict(torch.load(opts.other_params.arcface_path))
        
        self.face_pool_2 = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()
        
        self.set_requires_grad(False)
            
    def set_requires_grad(self, flag=True):
        for p in self.parameters():
            p.requires_grad = flag
    
    def extract_feats(self, x,clip_img=True):
        # breakpoint()
        if clip_img:
            x = un_norm_clip(x)
            x = TF.normalize(x, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        x = self.face_pool_1(x)  if x.shape[2]!=256 else  x # (1) resize to 256 if needed
        x = x[:, :, 35:223, 32:220]  # (2) Crop interesting region
        x = self.face_pool_2(x) # (3) resize to 112 to fit pre-trained model
        # breakpoint()
        x_feats = self.facenet(x, multi_scale=self.multiscale )
        
        # x_feats = self.facenet(x) # changed by sanoojan
        return x_feats

    

    def forward(self, y_hat, y,clip_img=True,return_seperate=False):
        n_samples = y.shape[0]
        y_feats_ms = self.extract_feats(y,clip_img=clip_img)  # Otherwise use the feature from there

        y_hat_feats_ms = self.extract_feats(y_hat,clip_img=clip_img)
        y_feats_ms = [y_f.detach() for y_f in y_feats_ms]
        
        loss_all = 0
        sim_improvement_all = 0
        seperate_sim=[]
        for y_hat_feats, y_feats in zip(y_hat_feats_ms, y_feats_ms):
 
            loss = 0
            sim_improvement = 0
            count = 0
            # lossess = []
            for i in range(n_samples):
                sim_target = y_hat_feats[i].dot(y_feats[i])
                sim_views = y_feats[i].dot(y_feats[i])

                seperate_sim.append(sim_target)
                loss += 1 - sim_target  # id loss
                sim_improvement +=  float(sim_target) - float(sim_views)
                count += 1
                
            
            loss_all += loss / count
            sim_improvement_all += sim_improvement / count
        if return_seperate:
            return loss_all, sim_improvement_all, seperate_sim
        return loss_all, sim_improvement_all, None

def uniform_on_device(r1, r2, shape, device):
    return (r1 - r2) * torch.rand(*shape, device=device) + r2

class LandmarkDetectionModel(nn.Module):
    def __init__(self):
        super(LandmarkDetectionModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(640, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.landmark_predictor = nn.Linear(128 * 32 * 32, 68 * 2)  # Adjust output size as needed

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        landmarks = self.landmark_predictor(x)
        return landmarks


class DDPM(pl.LightningModule):
    # classic DDPM with Gaussian diffusion, in image space
    def __init__(self,
                 unet_config,
                 timesteps=1000,
                 beta_schedule="linear",
                 loss_type="l2",
                 ckpt_path=None,
                 ignore_keys=[],
                 load_only_unet=False,
                 monitor="val/loss",
                 use_ema=True,
                 first_stage_key="image",
                 image_size=256,
                 channels=3,
                 log_every_t=100,
                 clip_denoised=True,
                 linear_start=1e-4,
                 linear_end=2e-2,
                 cosine_s=8e-3,
                 given_betas=None,
                 original_elbo_weight=0.,
                 v_posterior=0.,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
                 l_simple_weight=1.,
                 conditioning_key=None,
                 parameterization="eps",  # all assuming fixed variance schedules
                 scheduler_config=None,
                 use_positional_encodings=False,
                 learn_logvar=False,
                 logvar_init=0.,
                 u_cond_percent=0,
                 ):
        super().__init__()
        assert parameterization in ["eps", "x0"], 'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization
        print(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")
        self.cond_stage_model = None
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key
        self.image_size = image_size 
        self.channels = channels
        self.u_cond_percent=u_cond_percent
        self.use_positional_encodings = use_positional_encodings
        self.model = DiffusionWrapper(unet_config, conditioning_key)
        count_params(self.model, verbose=True)
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)

        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)

        self.loss_type = loss_type

        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)


    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                        2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start)
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool):
        model_out = self.model(x, t)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, return_intermediates=False):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        intermediates = [img]
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long),
                                clip_denoised=self.clip_denoised)
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)
        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, batch_size=16, return_intermediates=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size),
                                  return_intermediates=return_intermediates)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none') #-->
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def p_losses(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.model(x_noisy, t)

        loss_dict = {}
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        else:
            raise NotImplementedError(f"Paramterization {self.parameterization} not yet supported")

        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])

        log_prefix = 'train' if self.training else 'val'

        loss_dict.update({f'{log_prefix}/loss_simple': loss.mean()})
        loss_simple = loss.mean() * self.l_simple_weight

        loss_vlb = (self.lvlb_weights[t] * loss).mean()
        loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb})

        loss = loss_simple + self.original_elbo_weight * loss_vlb

        loss_dict.update({f'{log_prefix}/loss': loss})

        return loss, loss_dict

    def forward(self, x, *args, **kwargs):
        # b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        # assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x, t, *args, **kwargs)

    def get_input(self, batch, k):
        if k == "inpaint":
            x = batch['GT']
            mask = batch['inpaint_mask']
            inpaint = batch['inpaint_image']
            reference = batch['ref_imgs']
        else:
            x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        # x = rearrange(x, 'b h w c -> b c h w')
        x = x.to(memory_format=torch.contiguous_format).float()
        mask = mask.to(memory_format=torch.contiguous_format).float()
        inpaint = inpaint.to(memory_format=torch.contiguous_format).float()
        reference = reference.to(memory_format=torch.contiguous_format).float()
        return x,inpaint,mask,reference
    def get_input_ori(self, batch, k):
        if k == "inpaint":
            x = batch['GT']
            mask = batch['inpaint_mask']
            inpaint = batch['inpaint_image']
            reference = batch['ref_imgs']
        else:
            x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        # x = rearrange(x, 'b h w c -> b c h w')
        x = x.to(memory_format=torch.contiguous_format).float()
        mask = mask.to(memory_format=torch.contiguous_format).float()
        inpaint = inpaint.to(memory_format=torch.contiguous_format).float()
        reference = reference.to(memory_format=torch.contiguous_format).float()
        return x,inpaint,mask,reference

    def shared_step(self, batch):
        x = self.get_input(batch, self.first_stage_key)
        loss, loss_dict = self(x)
        return loss, loss_dict

    #----------dpo training step----------
    def training_step(self, batch, batch_idx):
        if not self.Reconstruct_initial:
            loss, loss_dict = self.shared_step(batch)       # original
        else:
            loss, loss_dict = self.shared_step_face(batch) # using dpo to train
        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)
        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)
        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return loss
    #----------dpo training step end----------

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        pass

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)
        
        # 在特定step进行测试
        # ⭐ 修复：使用标志位避免梯度累积时重复触发测试
        if (hasattr(self, 'test_during_training') and self.test_during_training and 
            hasattr(self, 'test_interval_steps') and 
            self.global_step > 0 and self.global_step % self.test_interval_steps == 0):
            # 检查是否已经在当前step测试过
            last_test_step = getattr(self, '_last_test_step', -1)
            if self.global_step != last_test_step:
                self._last_test_step = self.global_step
                self.run_test_during_training()
    

    def _get_rows_from_list(self, samples):
        n_imgs_per_row = len(samples)
        denoise_grid = rearrange(samples, 'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=2, sample=True, return_keys=None, **kwargs):
        log = dict()
        x = self.get_input(batch, self.first_stage_key)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        x = x.to(self.device)[:N]
        log["inputs"] = x

        # get diffusion row
        diffusion_row = list()
        x_start = x[:n_row]

        for t in range(self.num_timesteps):
            if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                t = t.to(self.device).long()
                noise = torch.randn_like(x_start)
                x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
                diffusion_row.append(x_noisy)

        log["diffusion_row"] = self._get_rows_from_list(diffusion_row)

        if sample:
            # get denoise row
            with self.ema_scope("Plotting"):
                samples, denoise_row = self.sample(batch_size=N, return_intermediates=True)

            log["samples"] = samples
            log["denoise_row"] = self._get_rows_from_list(denoise_row)

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.learn_logvar:
            params = params + [self.logvar]
        opt = torch.optim.AdamW(params, lr=lr)
        return opt


class LatentDiffusion(DDPM):
    """main class"""
    def __init__(self,
                 first_stage_config,
                 cond_stage_config,
                 num_timesteps_cond=None,
                 unet_config=None,             
                 ref_ckpt_path=None,      
                 dpo_beta=5000,
                 cond_stage_key="image",
                 cond_stage_trainable=False,
                 concat_mode=True,
                 cond_stage_forward=None,
                 conditioning_key=None,
                 scale_factor=1.0,
                 scale_by_std=False,
                 *args, **kwargs):
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        assert self.num_timesteps_cond <= kwargs['timesteps']
        # for backwards compatibility after implementation of DiffusionWrapper
        if conditioning_key is None:
            conditioning_key = 'concat' if concat_mode else 'crossattn'
        if cond_stage_config == '__is_unconditional__':
            conditioning_key = None
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        
        # 在调用父类 __init__ 之前，先提取 DPO 特有的参数
        # 这些参数父类 DDPM 不接受，需要先从 kwargs 中移除
        use_sft_loss = kwargs.pop('use_sft_loss', False)  # ⭐ 新增：是否使用SFT loss（False=使用DPO）
        use_auxiliary_losses = kwargs.pop('use_auxiliary_losses', False)
        aux_diffusion_weight = kwargs.pop('aux_diffusion_weight', 1.0)
        aux_id_loss_weight = kwargs.pop('aux_id_loss_weight', 0.3)
        aux_lpips_loss_weight = kwargs.pop('aux_lpips_loss_weight', 0.1)
        aux_lpips_multiscale = kwargs.pop('aux_lpips_multiscale', True)  # ⭐ 新增：LPIPS多尺度
        aux_reconstruct_weight = kwargs.pop('aux_reconstruct_weight', 1.0)  # ⭐ 新增：L2重构损失权重
        aux_reconstruct_ddim_steps = kwargs.pop('aux_reconstruct_ddim_steps', 4)
        dpo_loss_weight = kwargs.pop('dpo_loss_weight', 1.0)
        
        # ⭐ 可视化配置
        visualize_denoise_process = kwargs.pop('visualize_denoise_process', False)  # 是否可视化去噪过程
        visualize_interval_steps = kwargs.pop('visualize_interval_steps', 500)  # 每N步可视化一次
        
        # 训练时测试的配置（需要在调用父类之前移除，避免传递给父类）
        test_during_training = kwargs.pop('test_during_training', False)  # 是否在训练时进行测试
        test_interval_steps = kwargs.pop('test_interval_steps', 1000)  # 每N步测试一次
        test_num_samples = kwargs.pop('test_num_samples', 200)  # 测试时生成的样本数
        test_dataset_name = kwargs.pop('test_dataset_name', 'celeba')  # 测试数据集名称
        test_source_path = kwargs.pop('test_source_path', None)  # 测试源图像路径
        test_target_path = kwargs.pop('test_target_path', None)  # 测试目标图像路径
        test_source_mask_path = kwargs.pop('test_source_mask_path', None)  # 测试源mask路径
        test_target_mask_path = kwargs.pop('test_target_mask_path', None)  # 测试目标mask路径
        test_dataset_path = kwargs.pop('test_dataset_path', None)  # 测试数据集路径（用于FID）
        test_batch_size = kwargs.pop('test_batch_size', 4)  # 测试时的batch size
        test_guidance_scale = kwargs.pop('test_guidance_scale', 3.0)  # 测试时的guidance scale（参考inference_test_bench.sh，默认3.0）
        test_ddim_steps = kwargs.pop('test_ddim_steps', 50)  # 测试时的DDIM采样步数（参考inference_test_bench.sh，默认50）
        
        super().__init__(unet_config, conditioning_key=conditioning_key, *args, **kwargs)

        # 设置训练时测试的配置
        self.test_during_training = test_during_training
        self.test_interval_steps = test_interval_steps
        self.test_num_samples = test_num_samples
        self.test_dataset_name = test_dataset_name
        self.test_source_path = test_source_path
        self.test_target_path = test_target_path
        self.test_source_mask_path = test_source_mask_path
        self.test_target_mask_path = test_target_mask_path
        self.test_dataset_path = test_dataset_path
        self.test_batch_size = test_batch_size
        self.test_guidance_scale = test_guidance_scale
        self.test_ddim_steps = test_ddim_steps

        #---------------------DPO---------------------------------
        
        # ⭐ 优化：SFT 模式下不需要参考模型，节省显存
        if use_sft_loss:
            print(f"✓ SFT 模式：跳过参考模型加载（节省显存）")
            self.model_ref = None  # SFT 不需要参考模型
        else:
            print(f"DPO: 正在实例化参考模型 (Reference Model)...")
            self.model_ref = DiffusionWrapper(unet_config, conditioning_key)

            if ref_ckpt_path is None:
                print(f"DPO 警告: ref_ckpt_path 为 None。参考模型将使用随机权重。")
            else:
                print(f"DPO: 正在从 {ref_ckpt_path} 加载参考模型权重...")
                sd = torch.load(ref_ckpt_path, map_location="cpu")
                if "state_dict" in sd:
                    sd = sd["state_dict"]
                
                model_ref_sd = {k.replace('model.diffusion_model.', ''): v 
                                for k, v in sd.items() 
                                if k.startswith('model.diffusion_model.')}
                                
                self.model_ref.diffusion_model.load_state_dict(model_ref_sd, strict=True)
                print(f"DPO: 参考模型权重加载完毕。")

            #-------------------DPO结束---------------------------------

            self.model_ref.eval()
            self.model_ref.train = disabled_train
            for param in self.model_ref.parameters():
                param.requires_grad = False  # 参考模型不更新权重
            
        # 保存 DPO 配置（已经从 kwargs 中提取）
        self.use_sft_loss = use_sft_loss  # ⭐ 新增
        self.dpo_beta = dpo_beta
        self.use_auxiliary_losses = use_auxiliary_losses
        self.aux_diffusion_weight = aux_diffusion_weight
        self.aux_id_loss_weight = aux_id_loss_weight
        self.aux_lpips_loss_weight = aux_lpips_loss_weight
        self.aux_lpips_multiscale = aux_lpips_multiscale  # ⭐ 新增：LPIPS多尺度
        self.aux_reconstruct_weight = aux_reconstruct_weight  # ⭐ 新增：L2重构损失权重
        self.aux_reconstruct_ddim_steps = aux_reconstruct_ddim_steps
        self.dpo_loss_weight = dpo_loss_weight
        
        # ⭐ 可视化配置
        self.visualize_denoise_process = visualize_denoise_process
        self.visualize_interval_steps = visualize_interval_steps
        
        
        print(f"训练模式配置:")
        print(f"  - 使用 SFT Loss: {self.use_sft_loss} {'(有监督微调)' if self.use_sft_loss else '(DPO偏好优化)'}")
        print(f"  - DPO β参数: {self.dpo_beta}")
        print(f"  - DPO 损失权重: {self.dpo_loss_weight}")
        print(f"  - 是否启用辅助损失: {self.use_auxiliary_losses}")
        if self.use_auxiliary_losses or self.use_sft_loss:
            print(f"  - 扩散损失权重: {self.aux_diffusion_weight}")
            print(f"  - ID 损失权重: {self.aux_id_loss_weight}")
            print(f"  - L2 重构损失权重: {self.aux_reconstruct_weight}")
            print(f"  - LPIPS 感知损失权重: {self.aux_lpips_loss_weight}")
            if self.aux_lpips_loss_weight > 0:
                print(f"  - ⭐ LPIPS 多尺度: {self.aux_lpips_multiscale} {'(参考DPO: 512/256/128)' if self.aux_lpips_multiscale else '(单尺度)'}")
            print(f"  - DDIM 重构步数: {self.aux_reconstruct_ddim_steps}")
        print(f"  - ⭐ 可视化去噪过程: {self.visualize_denoise_process}")
        if self.visualize_denoise_process:
            print(f"  - 可视化间隔: 每 {self.visualize_interval_steps} 步")
     
        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        
        #check if other_params is present in cond_stage_config
        if hasattr(cond_stage_config, 'other_params'):
        
            self.clip_weight=cond_stage_config.other_params.clip_weight
            self.ID_weight=cond_stage_config.other_params.ID_weight
            self.Landmark_cond=cond_stage_config.other_params.Landmark_cond
            self.Landmarks_weight=cond_stage_config.other_params.Landmarks_weight
            if hasattr(cond_stage_config.other_params, 'Additional_config'):
                self.Reconstruct_initial=cond_stage_config.other_params.Additional_config.Reconstruct_initial
                self.Reconstruct_DDIM_steps=cond_stage_config.other_params.Additional_config.Reconstruct_DDIM_steps
                self.sampler=DDIMSampler(self)
                self.ID_loss_weight=cond_stage_config.other_params.Additional_config.ID_loss_weight
                self.LPIPS_loss_weight=cond_stage_config.other_params.Additional_config.LPIPS_loss_weight
                self.Landmark_loss_weight=cond_stage_config.other_params.Additional_config.Landmark_loss_weight
                if hasattr(cond_stage_config.other_params, 'multi_scale_ID'):
                    self.multi_scale_ID=cond_stage_config.other_params.multi_scale_ID   # True has an issue
                else:
                    self.multi_scale_ID=True  #this has an issue obtaining earlier layer from ID
                self.land_mark_id_seperate_layers=cond_stage_config.other_params.land_mark_id_seperate_layers
                self.sep_head_att=cond_stage_config.other_params.sep_head_att
                if hasattr(cond_stage_config.other_params, 'normalize'):
                    self.normalize=cond_stage_config.other_params.normalize  # normalizes the combintaion of ID and LPIPS loss
                else:
                    self.normalize=False
                if self.LPIPS_loss_weight>0:
                    self.lpips_loss = LPIPS(net_type='alex').to(self.device).eval()
                if hasattr(cond_stage_config.other_params, 'concat_feat'):
                    self.concat_feat=cond_stage_config.other_params.concat_feat
                else:
                    self.concat_feat=False
                if hasattr(cond_stage_config.other_params, 'stack_feat'):
                    self.stack_feat=cond_stage_config.other_params.stack_feat
                else:
                    self.stack_feat=False
                    
                if hasattr(cond_stage_config.other_params, 'weight_division'):
                    self.weight_division=cond_stage_config.other_params.weight_division
                else:
                    self.weight_division=True
                if hasattr(cond_stage_config.other_params, 'partial_training'):
                    self.partial_training=cond_stage_config.other_params.partial_training
                    self.trainable_keys=cond_stage_config.other_params.trainable_keys
                else:
                    self.partial_training=False
                if hasattr(cond_stage_config.other_params.Additional_config, 'Same_image_reconstruct'):
                    self.Same_image_reconstruct=cond_stage_config.other_params.Additional_config.Same_image_reconstruct
                else:
                    self.Same_image_reconstruct=False
                if hasattr(cond_stage_config.other_params.Additional_config, 'Target_CLIP_feat'):
                    self.Target_CLIP_feat=cond_stage_config.other_params.Additional_config.Target_CLIP_feat
                else:
                    self.Target_CLIP_feat=False
                if hasattr(cond_stage_config.other_params.Additional_config, 'Source_CLIP_feat'):
                    self.Source_CLIP_feat=cond_stage_config.other_params.Additional_config.Source_CLIP_feat
                else:
                    self.Source_CLIP_feat=False
                if hasattr(cond_stage_config.other_params.Additional_config, 'Multiple_ID_losses'):
                    self.Multiple_ID_losses=cond_stage_config.other_params.Additional_config.Multiple_ID_losses
                else:
                    self.Multiple_ID_losses=False
                if hasattr(cond_stage_config.other_params.Additional_config, 'use_3dmm'):  
                    self.use_3dmm=cond_stage_config.other_params.Additional_config.use_3dmm
                else:
                    self.use_3dmm=False
                if self.concat_feat:
                    self.concat_feat_proj=nn.Linear(768*2+136, 768)
                    # self.concat_feat_proj_out=nn.Linear(768, 768)
                if self.stack_feat:
                    pass
                    # self.stack_feat_proj=nn.Linear(768*2+136, 768)
                    # self.stack_feat_proj_out=nn.Linear(768, 768)
                    
            else:
                self.Reconstruct_initial=False
                self.Reconstruct_DDIM_steps=0
                
            self.update_weight=False  
                
        else:
            self.clip_weight=1
            self.ID_weight=0
            self.Landmark_cond=False
            self.Landmarks_weight=0
            self.Landmark_loss_weight=0
        if self.stack_feat:
            stacks=int(self.clip_weight>0)+int(self.ID_weight>0)+int(self.Landmarks_weight>0)
            self.learnable_vector = nn.Parameter(torch.randn((1,1,768)), requires_grad=True)
            self.other_learnable_vector = nn.Parameter(torch.randn((1,stacks-1,768)), requires_grad=True)
        else:
            self.learnable_vector = nn.Parameter(torch.randn((1,1,768)), requires_grad=True)
        if self.ID_weight>0:
            if self.multi_scale_ID:
                self.ID_proj_out=nn.Linear(200704, 768)
            else:
                self.ID_proj_out=nn.Linear(512, 768)
            self.instantiate_IDLoss(cond_stage_config)
            
        if self.Landmark_cond or self.Landmark_loss_weight>0:
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor("Other_dependencies/DLIB_landmark_det/shape_predictor_68_face_landmarks.dat")
            if self.Landmark_loss_weight>0:
                # from 640 channels 64 by 64 to 1 channel 64 by 64
                self.landmark_predictor=LandmarkDetectionModel()
                    
                        
            self.landmark_proj_out=nn.Linear(136, 768)
            if self.concat_feat:
                self.landmark_proj_out=nn.Identity()
            if self.stack_feat:
                self.landmark_proj_out=nn.Linear(136, 768)
        # total_devices = self.hparams.n_gpus * self.hparams.n_nodes
        # self.train_batches = len(self.train_dataloader()) // total_devices
        # self.train_reduce_steps = (self.hparams.epochs * self.train_batches) // (self.hparams.accumulate_grad_batches * 2) # for half of the time full trining with ID
        # self.change_weights=2/self.train_steps
        self.total_steps_in_epoch=0 # will be calculated inside training_step. Not known for now
        
        if cond_stage_config.target=="ldm.modules.encoders.modules.FrozenCLIPImageEmbedder":
            print("Using FrozenCLIPImageEmbedder")
            self.proj_out=nn.Linear(1024, 768)
        elif cond_stage_config.target=="ldm.modules.encoders.modules.FrozenCLIPEmbedder" and self.Source_CLIP_feat and self.Target_CLIP_feat:
            print("Using FrozenCLIPEmbedder")
            print("Using two projections for source and target")
            self.proj_out_source=nn.Linear(768, 768)
            self.proj_out_target=nn.Linear(768, 768)
            self.proj_out=nn.Identity()
        elif cond_stage_config.target=="ldm.modules.encoders.modules.FrozenCLIPEmbedder":
            print("Using FrozenCLIPEmbedder")
            self.proj_out=nn.Identity()
        
        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        except:
            self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))
        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)
        
        # self.TextEncoder=FrozenCLIPTextEmbedder()
        
        self.cond_stage_forward = cond_stage_forward
        self.clip_denoised = False
        self.bbox_tokenizer = None  

        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True

    def make_cond_schedule(self, ):
        self.cond_ids = torch.full(size=(self.num_timesteps,), fill_value=self.num_timesteps - 1, dtype=torch.long)
        ids = torch.round(torch.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)).long()
        self.cond_ids[:self.num_timesteps_cond] = ids

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        # only for very first batch
        if self.scale_by_std and self.current_epoch == 0 and self.global_step == 0 and batch_idx == 0 and not self.restarted_from_ckpt:
            assert self.scale_factor == 1., 'rather not use custom rescaling and std-rescaling simultaneously'
            # set rescale weight to 1./std of encodings
            print("### USING STD-RESCALING ###")
            x = super().get_input(batch, self.first_stage_key)
            x = x.to(self.device)
            encoder_posterior = self.encode_first_stage(x)
            z = self.get_first_stage_encoding(encoder_posterior).detach()
            del self.scale_factor
            self.register_buffer('scale_factor', 1. / z.flatten().std())
            print(f"setting self.scale_factor to {self.scale_factor}")
            print("### USING STD-RESCALING ###")

    def register_schedule(self,
                          given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        super().register_schedule(given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s)

        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule:
            self.make_cond_schedule()

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def instantiate_IDLoss(self, config):
        # Need to modify @sanoojan
        # if not self.cond_stage_trainable:
        model = IDLoss(config,multiscale=self.multi_scale_ID)
        self.face_ID_model = model.eval()
        self.face_ID_model.train = disabled_train
        for param in self.face_ID_model.parameters():
            param.requires_grad = False
            
        # else:
        #     assert config != '__is_first_stage__'
        #     assert config != '__is_unconditional__'
        #     model = IDLoss(config)
        #     self.face_ID_model = model.eval()  # currently not training and not training is better
    
    
    def instantiate_cond_stage(self, config):
        if not self.cond_stage_trainable:
            if config == "__is_first_stage__":
                print("Using first stage also as cond stage.")
                self.cond_stage_model = self.first_stage_model
            elif config == "__is_unconditional__":
                print(f"Training {self.__class__.__name__} as an unconditional model.")
                self.cond_stage_model = None
                # self.be_unconditional = True
            else:
                model = instantiate_from_config(config)
                self.cond_stage_model = model.eval()
                self.cond_stage_model.train = disabled_train
                for param in self.cond_stage_model.parameters():
                    param.requires_grad = False
        else:
            assert config != '__is_first_stage__'
            assert config != '__is_unconditional__'
            model = instantiate_from_config(config)
            self.cond_stage_model = model


    def _get_denoise_row_from_list(self, samples, desc='', force_no_decoder_quantization=False):
        denoise_row = []
        for zd in tqdm(samples, desc=desc):
            denoise_row.append(self.decode_first_stage(zd.to(self.device),
                                                            force_not_quantize=force_no_decoder_quantization))
        n_imgs_per_row = len(denoise_row)
        denoise_row = torch.stack(denoise_row)  # n_log_step, n_row, C, H, W
        denoise_grid = rearrange(denoise_row, 'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z

    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
                c = self.cond_stage_model.encode(c)  #--> c:[4,1,1024]
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode() 
            else:
                c = self.cond_stage_model(c)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c
    
    def conditioning_with_feat(self,x,landmarks=None,is_train=False,tar=None,tar_mask=None):
        c=0
        c2=0
        #find the model is in training or not
        is_train=self.training
        # self.train_reduce_steps=10000
        warmup_epoch=1
        # self.total_steps_in_epoch=7500
        reduce_weight_epochs=10
        if is_train and self.update_weight:
            if self.current_epoch<warmup_epoch:  #warmup
                self.clip_weight=1.0
                self.ID_weight=0.0
                if self.current_epoch<1:
                    self.total_steps_in_epoch=self.total_steps_in_epoch+1
                
                    self.train_reduce_steps=self.total_steps_in_epoch*(reduce_weight_epochs-warmup_epoch)
            else:
        
                self.clip_weight=(self.train_reduce_steps+self.total_steps_in_epoch-self.global_step)/(self.train_reduce_steps)
                self.ID_weight=1.0-self.clip_weight
                if self.clip_weight<0:
                    self.clip_weight=0.0
                    self.ID_weight=1.0
                    self.update_weight=False
                # print("weights:",self.clip_weight,self.ID_weight)
                    
        
        
        if self.clip_weight>0:
            
            if self.Source_CLIP_feat and self.Target_CLIP_feat and tar is not None:
                c_src=self.get_learned_conditioning(x)
                c_src = self.proj_out_source(c_src)
                
                tar1=tar*1.0
                tar1=un_norm(tar1)
                tar1=tar1.to(self.device)
                tar1=TF.normalize(tar1, mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
                # resize tar1 to 224
                tar1=TF.resize(tar1, (224,224))
                c = self.get_learned_conditioning(tar1) #-->c:[4,1,1024]
                c = self.proj_out_target(c) #-->c:[4,1,768]
                c=c_src+c
            
            elif self.Source_CLIP_feat and self.Target_CLIP_feat:
                c_src=self.get_learned_conditioning(x)
                c_src = self.proj_out_source(c_src)
                c=c_src
                
            
            elif self.use_3dmm and tar is None:


                
                self.models_3dmm.net_recon=self.models_3dmm.net_recon.to(x.device)
                c=self.models_3dmm.net_recon(x)
                c=self.dmm_proj_out(c)
                c=c.unsqueeze(1)
            elif self.use_3dmm and tar is not None:
                c_source=self.models_3dmm.net_recon(x)
                # c_src=self.dmm_proj_out(c_src)
                # c_src=c_src.unsqueeze(1)
                c=self.models_3dmm.net_recon(tar)
                # c_tar=self.dmm_proj_out(c_tar)
                # c_tar=c_tar.unsqueeze(1)
                c[:, :80]=c_source[:, :80]
                c=self.dmm_proj_out(c)
                c=c.unsqueeze(1)
        #         facemodel.split_coeff(output_coeff)
        #         id_coeffs = coeffs[:, :80]
        # exp_coeffs = coeffs[:, 80: 144]
        # tex_coeffs = coeffs[:, 144: 224]
        # angles = coeffs[:, 224: 227]
        # gammas = coeffs[:, 227: 254]
        # translations = coeffs[:, 254:]
                
                
                # c_src = self.proj_out(c_src)
            
            elif self.Target_CLIP_feat and tar is not None:
                tar1=tar*1.0
                tar1=un_norm(tar1)
                tar1=tar1.to(self.device)
                tar1=TF.normalize(tar1, mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
                # resize tar1 to 224
                tar1=TF.resize(tar1, (224,224))
                c = self.get_learned_conditioning(tar1) #-->c:[4,1,1024]
                c = self.proj_out(c) #-->c:[4,1,768]
                if self.normalize:
                    norm_coeff=c.norm(dim=2, keepdim=True)
            
            
            elif tar is not None:
                # tar1=tar*1.0
                # tar1=un_norm(tar1)
                # tar1=tar1.to(self.device)
                # # resize tar1 to 224
                # tar1=TF.resize(tar1, (224,224))

                # tar1=TF.normalize(tar1, mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
                # c=self.get_learned_conditioning(tar1)
                # c = self.proj_out(c)
                
                
                c_src=self.get_learned_conditioning(x)
                c_src = self.proj_out(c_src)
                
                # ExpressionTexts=['a photo of happy face','a photo of sad face','a photo of angry face','a photo of surprised face','a photo of disgusted face','a photo of scared face','a photo of neutral face']
                # ExpressionFeatures=self.TextEncoder(ExpressionTexts)
                # ExpressionFeatures=ExpressionFeatures[1]
                # ExpressionFeatures_normed=ExpressionFeatures/ExpressionFeatures.norm(dim=1, keepdim=True)
                # c_normed=c/c.norm(dim=2, keepdim=True)
                # c_normed=c_normed.squeeze(1)
                # c_src_normed=c_src/c_src.norm(dim=2, keepdim=True)
                # c_src_normed=c_src_normed.squeeze(1)
                
                # c_exp_basis=c_normed@ExpressionFeatures.T
                # c_src_exp_basis=c_src_normed@ExpressionFeatures.T
                # exp_coeff_diff=c_exp_basis-c_src_exp_basis
                # c_exp_weighted_features=exp_coeff_diff@ExpressionFeatures
                
                # c=c_src+c_exp_weighted_features.unsqueeze(1)
                
                c=c_src
                # c_exp_weighted_features=c_exp_basis@ExpressionFeatures
                
                 #-->c:[4,1,768]
                if self.normalize:
                    norm_coeff=c.norm(dim=2, keepdim=True)  
                
            else:
            
                c = self.get_learned_conditioning(x) #-->c:[4,1,1024]
                c = self.proj_out(c) #-->c:[4,1,768]
                if self.normalize:
                    norm_coeff=c.norm(dim=2, keepdim=True)
        if self.ID_weight>0:
            c2=self.face_ID_model.extract_feats(x)[0]
            c2 = self.ID_proj_out(c2) #-->c:[4,768]
            c2 = c2.unsqueeze(1) #-->c:[4,1,768]
            if self.normalize:
            #normalize c2
                c2 = c2*norm_coeff/F.normalize(c2, p=2, dim=2)
        
        if self.Landmark_cond==False:
            if self.weight_division:
                return (c*self.clip_weight+c2*self.ID_weight)/(self.clip_weight+self.ID_weight) 
            else:
                return c*self.clip_weight+c2*self.ID_weight  
        landmarks=landmarks.unsqueeze(1) if len(landmarks.shape)!=3 else landmarks
        if self.concat_feat:
            # concat c ,c2, landmarks
            conc=torch.cat([c,c2,landmarks],dim=-1)
            return self.concat_feat_proj(conc)
        if self.stack_feat:
            # stack c ,c2, landmarks
            conc=torch.cat([c,c2,landmarks],dim=-2)
            return conc
        if self.land_mark_id_seperate_layers or self.sep_head_att:
            if self.weight_division:
                c=(c*self.clip_weight+c2*self.ID_weight)/(self.clip_weight+self.ID_weight)
            else:
                c=c*self.clip_weight+c2*self.ID_weight
            conc=torch.cat([c,landmarks],dim=-1)
            return conc
        if self.weight_division:
            c=(c*self.clip_weight+c2*self.ID_weight+landmarks *self.Landmarks_weight)/(self.clip_weight+self.ID_weight+self.Landmarks_weight)
        else:
            c=c*self.clip_weight+c2*self.ID_weight+landmarks *self.Landmarks_weight
        # c = c.float()
        

        return c
    
    def conditioning_with_feat_given_features(self,landmarks=None,c2=None,c=None):
        # c=0
        # c2=0
          
        
        # if self.ID_weight>0:
        #     c2=self.face_ID_model.extract_feats(x)[0]
        #     c2 = self.ID_proj_out(c2) #-->c:[4,768]
        #     c2 = c2.unsqueeze(1) #-->c:[4,1,768]
        # if self.clip_weight>0:
        #     c = self.get_learned_conditioning(x) #-->c:[4,1,1024]
        #     c = self.proj_out(c) #-->c:[4,1,768]
        if self.Landmark_cond==False:
            return (c*self.clip_weight+c2*self.ID_weight)/(self.clip_weight+self.ID_weight)
        landmarks=landmarks.unsqueeze(1) if len(landmarks.shape)!=3 else landmarks
        c=(c*self.clip_weight+c2*self.ID_weight+landmarks *self.Landmarks_weight)/(self.clip_weight+self.ID_weight+self.Landmarks_weight)
        # c = c.float()
        

        return c

    def get_landmarks(self,x):
        def un_norm(x):
            return (x+1.0)/2.0
        
        if (self.Landmark_cond or self.Landmark_loss_weight>0) and x is not None:
            # pass
            # # Detect faces in an image
            #convert to 8bit image
            x=255.0*un_norm(x).permute(0,2,3,1).cpu().numpy()
            x=x.astype(np.uint8)
            Landmarks_all=[]    
            for i in range(len(x)):
                faces = self.detector(x[i],1)
                if len(faces)==0:
                    Landmarks_all.append(torch.zeros(1,136))
                    continue
                shape = self.predictor(x[i], faces[0])
                t = list(shape.parts())
                a = []
                for tt in t:
                    a.append([tt.x, tt.y])
                lm = np.array(a)
                lm = lm.reshape(1, 136)
                Landmarks_all.append(lm)
        Landmarks_all=np.concatenate(Landmarks_all,axis=0)
        Landmarks_all=torch.tensor(Landmarks_all).float().to(self.device)
        if self.Landmark_loss_weight>0 and self.Landmark_cond == False:
            return Landmarks_all
        Landmarks_all=self.landmark_proj_out(Landmarks_all)
        # normalize Landmarks_all
        
        return Landmarks_all

    def meshgrid(self, h, w):
        y = torch.arange(0, h).view(h, 1, 1).repeat(1, w, 1)
        x = torch.arange(0, w).view(1, w, 1).repeat(h, 1, 1)

        arr = torch.cat([y, x], dim=-1)
        return arr

    def delta_border(self, h, w):
        """
        :param h: height
        :param w: width
        :return: normalized distance to image border,
         wtith min distance = 0 at border and max dist = 0.5 at image center
        """
        lower_right_corner = torch.tensor([h - 1, w - 1]).view(1, 1, 2)
        arr = self.meshgrid(h, w) / lower_right_corner
        dist_left_up = torch.min(arr, dim=-1, keepdims=True)[0]
        dist_right_down = torch.min(1 - arr, dim=-1, keepdims=True)[0]
        edge_dist = torch.min(torch.cat([dist_left_up, dist_right_down], dim=-1), dim=-1)[0]
        return edge_dist

    def get_weighting(self, h, w, Ly, Lx, device):
        weighting = self.delta_border(h, w)
        weighting = torch.clip(weighting, self.split_input_params["clip_min_weight"],
                               self.split_input_params["clip_max_weight"], )
        weighting = weighting.view(1, h * w, 1).repeat(1, 1, Ly * Lx).to(device)

        if self.split_input_params["tie_braker"]:
            L_weighting = self.delta_border(Ly, Lx)
            L_weighting = torch.clip(L_weighting,
                                     self.split_input_params["clip_min_tie_weight"],
                                     self.split_input_params["clip_max_tie_weight"])

            L_weighting = L_weighting.view(1, 1, Ly * Lx).to(device)
            weighting = weighting * L_weighting
        return weighting

    def get_fold_unfold(self, x, kernel_size, stride, uf=1, df=1):  # todo load once not every time, shorten code
        """
        :param x: img of size (bs, c, h, w)
        :return: n img crops of size (n, bs, c, kernel_size[0], kernel_size[1])
        """
        bs, nc, h, w = x.shape

        # number of crops in image
        Ly = (h - kernel_size[0]) // stride[0] + 1
        Lx = (w - kernel_size[1]) // stride[1] + 1

        if uf == 1 and df == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold = torch.nn.Fold(output_size=x.shape[2:], **fold_params)

            weighting = self.get_weighting(kernel_size[0], kernel_size[1], Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h, w)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0], kernel_size[1], Ly * Lx))

        elif uf > 1 and df == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold_params2 = dict(kernel_size=(kernel_size[0] * uf, kernel_size[0] * uf),
                                dilation=1, padding=0,
                                stride=(stride[0] * uf, stride[1] * uf))
            fold = torch.nn.Fold(output_size=(x.shape[2] * uf, x.shape[3] * uf), **fold_params2)

            weighting = self.get_weighting(kernel_size[0] * uf, kernel_size[1] * uf, Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h * uf, w * uf)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0] * uf, kernel_size[1] * uf, Ly * Lx))

        elif df > 1 and uf == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold_params2 = dict(kernel_size=(kernel_size[0] // df, kernel_size[0] // df),
                                dilation=1, padding=0,
                                stride=(stride[0] // df, stride[1] // df))
            fold = torch.nn.Fold(output_size=(x.shape[2] // df, x.shape[3] // df), **fold_params2)

            weighting = self.get_weighting(kernel_size[0] // df, kernel_size[1] // df, Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h // df, w // df)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0] // df, kernel_size[1] // df, Ly * Lx))

        else:
            raise NotImplementedError

        return fold, unfold, normalization, weighting

    @torch.no_grad()
    def get_input(self, batch, k, return_first_stage_outputs=False, force_c_encode=False,
              cond_key=None, return_original_cond=False, bs=None, get_mask=False, 
              get_reference=False, get_landmarks_out=False, get_gt=False, get_inpaint=False,
              get_ref_raw=False):  # 新增参数
    
        # 1. 获取所有输入 (大小 B)
        x_w = batch['GT_w'].to(self.device, memory_format=torch.contiguous_format).float()
        x_l = batch['GT_l'].to(self.device, memory_format=torch.contiguous_format).float()
        GT_original = batch['GT'].to(self.device, memory_format=torch.contiguous_format).float()
        inpaint = batch['inpaint_image'].to(self.device, memory_format=torch.contiguous_format).float()
        mask = batch['inpaint_mask'].to(self.device, memory_format=torch.contiguous_format).float()
        reference = batch['ref_imgs'].to(self.device, memory_format=torch.contiguous_format).float()
        
        # 新增：获取未mask的参考图像（如果需要）
        if 'ref_img_raw' in batch:
            ref_raw = batch['ref_img_raw'].to(self.device, memory_format=torch.contiguous_format).float()
        else:
            ref_raw = None
        
        if bs is not None:
            x_w, x_l = x_w[:bs], x_l[:bs]
            GT_original = GT_original[:bs]
            inpaint, mask, reference = inpaint[:bs], mask[:bs], reference[:bs]
            if ref_raw is not None:
                ref_raw = ref_raw[:bs]  # ⭐ 修复：同时截断 ref_raw
        
        # 2. 编码共享条件
        z_inpaint = self.get_first_stage_encoding(self.encode_first_stage(inpaint)).detach()
        mask_resize = Resize([z_inpaint.shape[-1], z_inpaint.shape[-1]])(mask)
        
        # 3. 编码赢家和输家 (分别保持 [B, 9, 64, 64])
        z_w = self.get_first_stage_encoding(self.encode_first_stage(x_w)).detach()
        z_l = self.get_first_stage_encoding(self.encode_first_stage(x_l)).detach()
        
        z_new_w = torch.cat((z_w, z_inpaint, mask_resize), dim=1)  # [B, 9, 64, 64]
        z_new_l = torch.cat((z_l, z_inpaint, mask_resize), dim=1)  # [B, 9, 64, 64]
        
        # 4. 获取共享条件 [B, 1, 768]
        if self.Landmark_cond or self.Landmark_loss_weight > 0:
            landmarks = self.get_landmarks(GT_original)
        else:
            landmarks = None
        
        if self.model.conditioning_key is not None:
            # ⭐ 修复：应该使用带mask的 reference（ref_imgs）作为条件
            # 原因：只保留核心人脸区域，特征更纯净，减少干扰
            # reference: B图像（带mask，只保留核心人脸）
            # ref_raw: B图像（未mask，完整图像）- 仅用于ID损失
            xc = reference  # ← 修复：使用带mask的版本
            if not self.cond_stage_trainable or force_c_encode:
                c_shared = self.conditioning_with_feat(xc, landmarks=landmarks).float()
            else:
                c_shared = xc
        else:
            c_shared = None
        
        # 5. ⭐ 输出：分别返回赢家、输家、条件（都是 B）
        out = [z_new_w, z_new_l, c_shared]
        
        # 6. 可选返回值（都保持 B）
        if return_first_stage_outputs:
            xrec_w = self.decode_first_stage(z_w)
            xrec_l = self.decode_first_stage(z_l)
            out.extend([x_w, x_l, xrec_w, xrec_l])
        if return_original_cond:
            out.append(reference)
        if get_mask:
            out.append(mask)
        if get_reference:
            out.append(reference)
        if get_landmarks_out:
            out.append(landmarks)
        if get_gt:
            out.append(GT_original)
        if get_inpaint:
            out.append(inpaint)
        if get_ref_raw:
            out.append(ref_raw)  # 新增：返回未mask的参考图像
        
        return out
  
        
    @torch.no_grad()
    def decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()

        z = 1. / self.scale_factor * z

        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                uf = self.split_input_params["vqf"]
                bs, nc, h, w = z.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(z, ks, stride, uf=uf)

                z = unfold(z)  # (bn, nc * prod(**ks), L)
                # 1. Reshape to img shape
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                # 2. apply model loop over last dim
                if isinstance(self.first_stage_model, VQModelInterface):
                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i],
                                                                 force_not_quantize=predict_cids or force_not_quantize)
                                   for i in range(z.shape[-1])]
                else:

                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i])
                                   for i in range(z.shape[-1])]

                o = torch.stack(output_list, axis=-1)  # # (bn, nc, ks[0], ks[1], L)
                o = o * weighting
                # Reverse 1. reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization  # norm is shape (1, 1, h, w)
                return decoded
            else:
                if isinstance(self.first_stage_model, VQModelInterface):
                    return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
                else:
                    return self.first_stage_model.decode(z)

        else:
            if isinstance(self.first_stage_model, VQModelInterface):
                return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
            else:
                if self.first_stage_key=='inpaint':
                    return self.first_stage_model.decode(z[:,:4,:,:])
                else:
                    return self.first_stage_model.decode(z)

   

    # same as above but without decorator
    def differentiable_decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()

        z = 1. / self.scale_factor * z

        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                uf = self.split_input_params["vqf"]
                bs, nc, h, w = z.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(z, ks, stride, uf=uf)

                z = unfold(z)  # (bn, nc * prod(**ks), L)
                # 1. Reshape to img shape
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                # 2. apply model loop over last dim
                if isinstance(self.first_stage_model, VQModelInterface):  
                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i],
                                                                 force_not_quantize=predict_cids or force_not_quantize)
                                   for i in range(z.shape[-1])]
                else:

                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i])
                                   for i in range(z.shape[-1])]

                o = torch.stack(output_list, axis=-1)  # # (bn, nc, ks[0], ks[1], L)
                o = o * weighting
                # Reverse 1. reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization  # norm is shape (1, 1, h, w)
                return decoded
            else:
                if isinstance(self.first_stage_model, VQModelInterface):
                    return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
                else:
                    return self.first_stage_model.decode(z)

        else:
            if isinstance(self.first_stage_model, VQModelInterface):
                return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
            else:
                return self.first_stage_model.decode(z)

    @torch.no_grad()
    def encode_first_stage(self, x):
        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                df = self.split_input_params["vqf"]
                self.split_input_params['original_image_size'] = x.shape[-2:]
                bs, nc, h, w = x.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(x, ks, stride, df=df)
                z = unfold(x)  # (bn, nc * prod(**ks), L)
                # Reshape to img shape
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                output_list = [self.first_stage_model.encode(z[:, :, :, :, i])
                               for i in range(z.shape[-1])]

                o = torch.stack(output_list, axis=-1)
                o = o * weighting

                # Reverse reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization
                return decoded

            else:
                return self.first_stage_model.encode(x)
        else:
            return self.first_stage_model.encode(x)

    def shared_step(self, batch, **kwargs):
        x, c ,landmarks= self.get_input(batch, self.first_stage_key,get_landmarks_out=True)
        loss = self(x, c,landmarks=landmarks)
        return loss
    
    def shared_step_face(self, batch, **kwargs):
        # ⭐ 解包：z_w, z_l, c 都是 [B, ...]
       # 修改后：添加 force_c_encode=True 和 get_ref_raw=True
        outputs = self.get_input(batch, self.first_stage_key, 
                        force_c_encode=True,  # ← 强制编码条件
                        get_landmarks_out=True, 
                        get_reference=True, 
                        get_gt=True,
                        get_ref_raw=True)  # ← 新增：获取未mask的参考图像
        
        z_w = outputs[0]          # [B, 9, 64, 64] - 赢家潜变量
        z_l = outputs[1]          # [B, 9, 64, 64] - 输家潜变量
        c = outputs[2]            # [B, 1, 768] - 条件
        reference = outputs[3]    # [B, 3, 224, 224] - 参考图像（已mask）
        landmarks = outputs[4]    # landmarks（如果需要）
        GT_w = outputs[5]         # [B, 3, H, W] - 赢家的 GT 图像
        ref_raw = outputs[6]      # [B, 3, 224, 224] - 参考图像（未mask）⭐ 新增
        
        # 调用 forward_face，传递所有需要的参数
        loss = self.forward_face(z_w, z_l, c, 
                               landmarks=landmarks,
                               reference=reference,
                               ref_raw=ref_raw,  # ← 新增
                               GT_w=GT_w)
        return loss

    def forward(self, x, c,landmarks=None, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        self.u_cond_prop=random.uniform(0, 1)
        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                
                c=self.conditioning_with_feat(c,landmarks=landmarks)
                    
            if self.shorten_cond_schedule:  # TODO: drop this option
                tc = self.cond_ids[t].to(self.device)
                c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))

        if self.u_cond_prop<self.u_cond_percent and self.training :
            if self.land_mark_id_seperate_layers or self.sep_head_att:
                conc=self.learnable_vector.repeat(x.shape[0],1,1)
                # concat c, landmarks
                landmarks=landmarks.unsqueeze(1) if len(landmarks.shape)!=3 else landmarks
                conc=torch.cat([conc,landmarks],dim=-1)
                return self.p_losses(x, conc, t, *args, **kwargs)
            return self.p_losses(x, self.learnable_vector.repeat(x.shape[0],1,1), t, *args, **kwargs)
        else:  #x:[4,9,64,64] c:[4,1,768] x: img,inpaint_img,mask after first stage c:clip embedding
            return self.p_losses(x, c, t, *args, **kwargs)
    
    def forward_face(self, z_w, z_l, c, landmarks=None, reference=None, ref_raw=None, GT_w=None, 
                    *args, **kwargs):
        """
        ⭐ 直接接收三个 B 批次的张量
        z_w: [B, 9, 64, 64] - 赢家潜变量
        z_l: [B, 9, 64, 64] - 输家潜变量
        c: [B, 1, 768]      - 共享条件
        reference: [B, 3, 224, 224] - 参考图像（已mask，用于条件）
        ref_raw: [B, 3, 224, 224] - 参考图像（未mask，用于 ID 损失）⭐ 新增
        GT_w: [B, 3, H, W] - 赢家的 GT 图像（用于感知损失）
        """
        # 1. 获取批次大小
        B = z_w.shape[0]
        
        # 2. 采样共享时间步
        t = torch.randint(0, self.num_timesteps, (B,), device=self.device).long()
        
        # 3. ⭐ 传递额外参数到 p_losses_dpo
        if self.use_sft_loss:
            return self.p_loss_sft(z_w, z_l, c, t, 
                                reference=reference,
                                ref_raw=ref_raw,  # ← 新增
                                GT_w=GT_w, 
                                landmarks=landmarks,
                                *args, **kwargs)    
        else:
            return self.p_losses_dpo(z_w, z_l, c, t, 
                                reference=reference,
                                ref_raw=ref_raw,  # ← 新增
                                GT_w=GT_w, 
                                landmarks=landmarks,
                                *args, **kwargs)
                

    def _rescale_annotations(self, bboxes, crop_coordinates):  # TODO: move to dataset
        def rescale_bbox(bbox):
            x0 = clamp((bbox[0] - crop_coordinates[0]) / crop_coordinates[2])
            y0 = clamp((bbox[1] - crop_coordinates[1]) / crop_coordinates[3])
            w = min(bbox[2] / crop_coordinates[2], 1 - x0)
            h = min(bbox[3] / crop_coordinates[3], 1 - y0)
            return x0, y0, w, h

        return [rescale_bbox(b) for b in bboxes]

    def apply_model(self, x_noisy, t, cond, return_ids=False,return_features=False):

        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn' # -->c_crossattn
            cond = {key: cond}

        if hasattr(self, "split_input_params"):
            assert len(cond) == 1  # todo can only deal with one conditioning atm
            assert not return_ids  
            ks = self.split_input_params["ks"]  # eg. (128, 128)
            stride = self.split_input_params["stride"]  # eg. (64, 64)

            h, w = x_noisy.shape[-2:]

            fold, unfold, normalization, weighting = self.get_fold_unfold(x_noisy, ks, stride)

            z = unfold(x_noisy)  # (bn, nc * prod(**ks), L)
            # Reshape to img shape
            z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )
            z_list = [z[:, :, :, :, i] for i in range(z.shape[-1])]

            if self.cond_stage_key in ["image", "LR_image", "segmentation",
                                       'bbox_img'] and self.model.conditioning_key:  # todo check for completeness
                c_key = next(iter(cond.keys()))  # get key
                c = next(iter(cond.values()))  # get value
                assert (len(c) == 1)  # todo extend to list with more than one elem
                c = c[0]  # get element

                c = unfold(c)
                c = c.view((c.shape[0], -1, ks[0], ks[1], c.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                cond_list = [{c_key: [c[:, :, :, :, i]]} for i in range(c.shape[-1])]

            elif self.cond_stage_key == 'coordinates_bbox':
                assert 'original_image_size' in self.split_input_params, 'BoudingBoxRescaling is missing original_image_size'

                # assuming padding of unfold is always 0 and its dilation is always 1
                n_patches_per_row = int((w - ks[0]) / stride[0] + 1)
                full_img_h, full_img_w = self.split_input_params['original_image_size']
                # as we are operating on latents, we need the factor from the original image size to the
                # spatial latent size to properly rescale the crops for regenerating the bbox annotations
                num_downs = self.first_stage_model.encoder.num_resolutions - 1
                rescale_latent = 2 ** (num_downs)

                # get top left positions of patches as conforming for the bbbox tokenizer, therefore we
                # need to rescale the tl patch coordinates to be in between (0,1)
                tl_patch_coordinates = [(rescale_latent * stride[0] * (patch_nr % n_patches_per_row) / full_img_w,
                                         rescale_latent * stride[1] * (patch_nr // n_patches_per_row) / full_img_h)
                                        for patch_nr in range(z.shape[-1])]

                # patch_limits are tl_coord, width and height coordinates as (x_tl, y_tl, h, w)
                patch_limits = [(x_tl, y_tl,
                                 rescale_latent * ks[0] / full_img_w,
                                 rescale_latent * ks[1] / full_img_h) for x_tl, y_tl in tl_patch_coordinates]
                # patch_values = [(np.arange(x_tl,min(x_tl+ks, 1.)),np.arange(y_tl,min(y_tl+ks, 1.))) for x_tl, y_tl in tl_patch_coordinates]

                # tokenize crop coordinates for the bounding boxes of the respective patches
                patch_limits_tknzd = [torch.LongTensor(self.bbox_tokenizer._crop_encoder(bbox))[None].to(self.device)
                                      for bbox in patch_limits]  # list of length l with tensors of shape (1, 2)
                print(patch_limits_tknzd[0].shape)
                # cut tknzd crop position from conditioning
                assert isinstance(cond, dict), 'cond must be dict to be fed into model'
                cut_cond = cond['c_crossattn'][0][..., :-2].to(self.device)

                adapted_cond = torch.stack([torch.cat([cut_cond, p], dim=1) for p in patch_limits_tknzd])
                adapted_cond = rearrange(adapted_cond, 'l b n -> (l b) n')
                adapted_cond = self.get_learned_conditioning(adapted_cond)
                adapted_cond = rearrange(adapted_cond, '(l b) n d -> l b n d', l=z.shape[-1])

                cond_list = [{'c_crossattn': [e]} for e in adapted_cond]

            else:
                cond_list = [cond for i in range(z.shape[-1])]  # Todo make this more efficient

            # apply model by loop over crops
            output_list = [self.model(z_list[i], t, **cond_list[i]) for i in range(z.shape[-1])]
            assert not isinstance(output_list[0],
                                  tuple)  # todo cant deal with multiple model outputs check this never happens

            o = torch.stack(output_list, axis=-1)
            o = o * weighting
            # Reverse reshape to img shape
            o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
            # stitch crops together
            x_recon = fold(o) / normalization

        else:
            x_recon = self.model(x_noisy, t, **cond, return_features=return_features)
        if return_features:
            return x_recon
        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart) / \
               extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.
        This term can't be optimized, as it only depends on the encoder.
        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = torch.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0)
        return mean_flat(kl_prior) / np.log(2.0)

    def p_losses(self, x_start, cond, t, noise=None, ):
        if self.first_stage_key == 'inpaint':
            # x_start=x_start[:,:4,:,:]
            noise = default(noise, lambda: torch.randn_like(x_start[:,:4,:,:]))
            x_noisy = self.q_sample(x_start=x_start[:,:4,:,:], t=t, noise=noise)
            x_noisy = torch.cat((x_noisy,x_start[:,4:,:,:]),dim=1)
        else:
            noise = default(noise, lambda: torch.randn_like(x_start))
            x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        self.logvar = self.logvar.to(self.device)
        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3)) #??
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})
        
        # ⭐ 修复阻塞问题：commit=False，避免每个batch都等待上传
        if wandb.run is not None:
            wandb.log(loss_dict, commit=False)
        return loss, loss_dict
    def p_loss_sft(self, z_start_w, z_start_l, cond, t, reference=None, ref_raw=None, GT_w=None, landmarks=None):
        """
        SFT损失函数 - 参考论文公式
        L = λ_id * L_id + λ_DM * L_DM + λ_rec * L_rec + λ_lpips * L_lpips
        
        参数说明（换脸任务）:
        z_start_w: GT潜变量 (B, 9, 64, 64) - [z0_A, z_inp, m]
                   - z0_A (0:4): A图像的latent（换脸后的正确结果 - 真正的GT）
                   - z_inp (4:8): D图像的latent（提供姿态）
                   - m (8:9): mask
        z_start_l: 不使用（为了兼容接口保留，SFT不需要loser）
        cond: 条件 (B, 1, 768) 
        t: 时间步 (B,) - 论文中使用t=999进行one-step训练
        reference: B图像在像素空间（已mask，用于条件）
        ref_raw: B图像在像素空间（未mask，用于ID损失）⭐ 新增
        GT_w: A图像在像素空间（真正的GT，用于重构损失）
        
        换脸任务概念：
        - A = 真正的GT（换脸后的正确结果）
        - B = Source（提供身份）
        - D = Target（提供姿态）
        - 目标：学习 (D + B的条件) → A 的映射
    
        """
        loss_dict = {}
        prefix = 'train' if self.training else 'val'
        
        
        # ========== 1️⃣ Diffusion Loss (L_DM) - 主要损失 ==========
        # 论文: L_DM = E[||ε - ε_θ(z_t, c, t)||_2]
        
        # 采样噪声
        noise = torch.randn_like(z_start_w[:, :4, :, :])
        
        # 对GT（A）加噪
        z0_A = z_start_w[:, :4, :, :]  # A图像的latent
        z_t = self.q_sample(x_start=z0_A, t=t, noise=noise)
        
        # 拼接输入（D）和mask
        z_noisy = torch.cat([z_t, z_start_w[:, 4:, :, :]], dim=1)  # (B, 9, 64, 64)
        
        # 模型预测噪声
        pred_noise = self.apply_model(z_noisy, t, cond)
        
        # 计算Diffusion损失
        loss_DM = self.get_loss(pred_noise, noise, mean=True)
        loss_dict[f'{prefix}/loss_DM'] = loss_DM
        
        # 主损失（根据配置的权重）
        loss_total = self.aux_diffusion_weight * loss_DM
        
        # 采样一个新的 t_new（固定为最大时间步，用于重建损失计算）
        # 从最大噪声状态开始去噪，测试完整的去噪能力
        # t_new = torch.randint(
        #     self.num_timesteps - 1, 
        #     self.num_timesteps, 
        #     (z_start_w.shape[0],), 
        #     device=self.device
        # ).long()
        
        # # 使用 t_new 重新加噪（从最大噪声开始）
        # z_t_new = self.q_sample(x_start=z0_A, t=t_new, noise=noise)
        
        # ========== 2️⃣ ID Loss (L_id) + Reconstruction Loss (L_rec) + LPIPS Loss ==========
       
                    
                else:
                    # 如果没有sampler，回退到单步预测
                    print("警告: 未找到sampler，回退到单步预测")
                    pred_x0 = self.predict_start_from_noise(z_t, t, pred_noise)
                    x_pred = self.differentiable_decode_first_stage(pred_x0)
                    decoded_pred_x_0 = [x_pred]  # 只有一个步骤
            
            # ⭐ 应用（ID、重构、感知损失
            # 2.1 ID Loss - L_id = 1 - cos(e_ID_A1, e_ID_Ã)
            # ⭐ 修改：与原始代码一致，对所有中间步骤计算并平均
            if self.aux_id_loss_weight > 0 and hasattr(self, 'face_ID_model'):
                # ⭐ 直接使用已mask的参考图像 reference（数据加载时已处理）
                if reference is not None:
                    # 准备Source图像B（从CLIP归一化转为标准归一化）
                    source_normalized = un_norm_clip(reference)
                    source_normalized = TF.normalize(source_normalized,
                                                    mean=[0.5, 0.5, 0.5],
                                                    std=[0.5, 0.5, 0.5])
                else:
                    source_normalized = None
            
            if self.aux_id_loss_weight > 0 and source_normalized is not None and hasattr(self, 'face_ID_model'):
                
                # 准备mask（只在人脸区域计算）
                masks = 1 - TF.resize(z_start_w[:, 8, :, :],
                                     (decoded_pred_x_0[0].shape[2], decoded_pred_x_0[0].shape[3]))
                
                # ⭐ 只用最后一步计算ID损失（节省显存和计算）
                # 如果想要所有步骤，在显存充足时再改回来
                x_pred_final = decoded_pred_x_0[-1]  # 只取最后一步
                x_pred_masked = x_pred_final * masks.unsqueeze(1)
                
                # 计算ID损失（余弦相似度损失）
                loss_id, sim_imp, _ = self.face_ID_model(
                    x_pred_masked, source_normalized, clip_img=False
                )
                
                # ⭐ 应用时间步权重
                loss_total += self.aux_id_loss_weight * loss_id 
                loss_dict[f'{prefix}/loss_id'] = loss_id
                loss_dict[f'{prefix}/sim_imp'] = sim_imp
            
            # 2.2 L2 Reconstruction Loss - L_rec = ||A - Ã||_2^2
            # 只用最后一步计算重构损失
            if self.aux_reconstruct_weight > 0 and GT_w is not None:
                # 计算L2重构损失
                loss_rec_l2 = self.get_loss(x_pred, GT_w, mean=True)
                
                # ⭐ 应用时间步权重
                loss_total += self.aux_reconstruct_weight * loss_rec_l2
                loss_dict[f'{prefix}/loss_rec_l2'] = loss_rec_l2
            
            # 2.3 ⭐ LPIPS Perceptual Loss - L_lpips = LPIPS(A, Ã)
            # ⭐ 只用最后一步（节省显存），多尺度
            if self.aux_lpips_loss_weight > 0 and GT_w is not None and hasattr(self, 'lpips_loss'):
                if self.aux_lpips_multiscale:
                    # 只用最后一步 × 多尺度
                    loss_lpips = 0
                    for scale_idx in range(3):  # 3个尺度：512, 256, 128
                        scale_size = 512 // (2 ** scale_idx)
                        loss_lpips_scale = self.lpips_loss(
                            F.adaptive_avg_pool2d(x_pred, (scale_size, scale_size)),
                            F.adaptive_avg_pool2d(GT_w, (scale_size, scale_size))
                        ).mean()
                        loss_lpips += loss_lpips_scale
                        loss_dict[f'{prefix}/loss_lpips_scale_{scale_idx}'] = loss_lpips_scale
                else:
                    # 单尺度：只用最后一步
                    loss_lpips = self.lpips_loss(x_pred, GT_w).mean()
                
                # ⭐ 应用时间步权重
                loss_total += self.aux_lpips_loss_weight * loss_lpips 
                loss_dict[f'{prefix}/loss_lpips'] = loss_lpips
                
        
        # ========== 总损失 ==========
        # 论文: L = λ_id * L_id + λ_DM * L_DM + λ_rec * L_rec + λ_lpips * L_lpips
        # ⭐ 新增：时间步权重机制，当t小时给予更大权重
        loss_dict[f'{prefix}/loss'] = loss_total
        
        if wandb.run is not None:
            wandb.log(loss_dict, commit=False)
        
        return loss_total, loss_dict

    def p_losses_dpo(self, z_start_w, z_start_l, cond, t, reference=None, ref_raw=None, GT_w=None, landmarks=None):
        """
        计算 Diffusion DPO 损失 + 可选的辅助损失

        参数:
        z_start_w: "赢家" 潜变量 (B, 9, 64, 64) - [z0_w, z_inp, m]，A
        z_start_l: "输家" 潜变量 (B, 9, 64, 64) - [z0_l, z_inp, m],E
        cond: 共享条件 (B, 1, 768) - (来自 ref_imgs)
        t: 共享时间步 (B,)
        reference: 参考图像（已mask，用于条件）
        ref_raw: 参考图像（未mask，用于 ID 损失）⭐ 新增
        GT_w:实际是B,伪造target
        landmarks: B的landmarks
        """
        # (D1) 修正：分别采样独立噪声 epsilon_w 和 epsilon_l
        noise_w = torch.randn_like(z_start_w[:, :4, :, :])
        noise_l = torch.randn_like(z_start_l[:, :4, :, :])

        # 赢家：对 z0_w 加噪
        z_t_w = self.q_sample(x_start=z_start_w[:, :4, :, :], t=t, noise=noise_w)
        z_noisy_w = torch.cat((z_t_w, z_start_w[:, 4:, :, :]), dim=1) 

        # 输家：对 z0_l 加噪
        #如果dpo权重不等于0
        z_t_l = self.q_sample(x_start=z_start_l[:, :4, :, :], t=t, noise=noise_l)
        z_noisy_l = torch.cat((z_t_l, z_start_l[:, 4:, :, :]), dim=1) 
        
        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        # (D2) 获取策略模型(policy)和参考模型(ref)的噪声预测
        c_dict = {__conditioning_keys__[self.model.conditioning_key]: [cond]}
        
        pred_noise_policy_w = self.apply_model(z_noisy_w, t, cond)
        pred_noise_policy_l = self.apply_model(z_noisy_l, t, cond)

        # ⭐ 安全检查：确保参考模型存在（DPO模式才需要）
        if self.model_ref is None:
            raise RuntimeError("参考模型未初始化！p_losses_dpo 需要参考模型，请检查配置。")
        
        with torch.no_grad():
            pred_noise_ref_w = self.model_ref(z_noisy_w, t, **c_dict)
            pred_noise_ref_l = self.model_ref(z_noisy_l, t, **c_dict)

        
        # (D3) 修正：计算 Loss 时，target 必须对应各自的 noise
        # 赢家比较 pred_w 和 noise_w
        loss_policy_w = self.get_loss(pred_noise_policy_w, noise_w, mean=False).mean(dim=[1, 2, 3])
        loss_ref_w    = self.get_loss(pred_noise_ref_w,    noise_w, mean=False).mean(dim=[1, 2, 3])
        
        # 输家比较 pred_l 和 noise_l
        loss_policy_l = self.get_loss(pred_noise_policy_l, noise_l, mean=False).mean(dim=[1, 2, 3])
        loss_ref_l    = self.get_loss(pred_noise_ref_l,    noise_l, mean=False).mean(dim=[1, 2, 3])
        # (D4) 构建优势项 A_theta_ref(t)
        A_theta_ref = (loss_policy_w - loss_policy_l) - (loss_ref_w - loss_ref_l).detach()
        
        # (D5) 最终 Diffusion-DPO 损失
        logits = -self.dpo_beta * A_theta_ref
        loss_dpo = -torch.nn.functional.logsigmoid(logits).mean()
        
        # 记录 DPO 损失
        loss_dict.update({f'{prefix}/loss_dpo': loss_dpo})
        loss_dict.update({f'{prefix}/A_theta_ref_mean': A_theta_ref.mean()})
        loss_dict.update({f'{prefix}/loss_policy_w_mean': loss_policy_w.mean()})
        loss_dict.update({f'{prefix}/loss_policy_l_mean': loss_policy_l.mean()})
        
        # ========== 辅助损失部分 ==========
        loss_aux_total = 0.0
        
        if self.use_auxiliary_losses:
            # 1️⃣ 扩散重构损失（对赢样本）
            if self.aux_diffusion_weight > 0:
                loss_diffusion = self.get_loss(pred_noise_policy_w, noise_w, mean=True)
                loss_aux_total += self.aux_diffusion_weight * loss_diffusion
                loss_dict.update({f'{prefix}/loss_aux_diffusion': loss_diffusion})
            
            # 2️⃣ & 3️⃣ ID 损失 + 感知损失（需要多步 DDIM 重构）
            if (self.aux_id_loss_weight > 0 or self.aux_lpips_loss_weight > 0) and hasattr(self, 'sampler'):
                # 参考原始 REFace 的重构流程
                # Step 1: 对赢样本加最大噪声
                t_new = torch.randint(self.num_timesteps-1, self.num_timesteps, 
                                     (z_start_w.shape[0],), device=self.device).long()
                z_noisy_rec = self.q_sample(x_start=z_start_w[:, :4, :, :], t=t_new, noise=noise_w)
                z_noisy_rec = torch.cat((z_noisy_rec, z_start_w[:, 4:, :, :]), dim=1)
                
                # Step 2: DDIM 多步去噪重构
                ddim_steps = self.aux_reconstruct_ddim_steps
                n_samples = z_noisy_rec.shape[0]
                shape = (4, 64, 64)
                ddim_eta = 0.0
                
                # 使用当前条件进行重构（不需要 flip）
                samples_ddim, sample_intermediates = self.sampler.sample_train(
                    S=ddim_steps,
                    conditioning=cond,
                    batch_size=n_samples,
                    shape=shape,
                    verbose=False,
                    unconditional_guidance_scale=5,
                    unconditional_conditioning=None,
                    eta=ddim_eta,
                    x_T=z_noisy_rec,
                    t=t_new,
                    test_model_kwargs=None
                )
                
                # Step 3: 获取每步的 pred_x0 并解码到图像空间
                other_pred_x_0 = sample_intermediates['pred_x0']
                decoded_pred_x_0 = []
                for i in range(len(other_pred_x_0)):
                    decoded = self.differentiable_decode_first_stage(other_pred_x_0[i])
                    decoded_pred_x_0.append(decoded)
                
                # Step 4: 计算 ID 损失
                if self.aux_id_loss_weight > 0 and hasattr(self, 'face_ID_model'):
                    # ⭐ 优先使用未mask的参考图像 ref_raw
                    source_for_id = ref_raw if ref_raw is not None else reference
                    
                    if source_for_id is not None:
                        # 准备参考图像（从CLIP归一化转为标准归一化）
                        reference_normalized = un_norm_clip(source_for_id)
                        reference_normalized = TF.normalize(reference_normalized, 
                                                           mean=[0.5, 0.5, 0.5], 
                                                           std=[0.5, 0.5, 0.5])
                    else:
                        reference_normalized = None
                
                if self.aux_id_loss_weight > 0 and reference_normalized is not None and hasattr(self, 'face_ID_model'):
                    
                    # 准备 mask
                    masks = 1 - TF.resize(z_start_w[:, 8, :, :], 
                                         (decoded_pred_x_0[0].shape[2], decoded_pred_x_0[0].shape[3]))
                    
                    # 对每步的重构结果计算 ID 损失
                    ID_Losses = []
                    for step, x_pred in enumerate(decoded_pred_x_0):
                        x_pred_masked = x_pred * masks.unsqueeze(1)
                        ID_loss_step, sim_imp, _ = self.face_ID_model(
                            x_pred_masked, reference_normalized, clip_img=False
                        )
                        ID_Losses.append(ID_loss_step)
                        loss_dict.update({f'{prefix}/loss_aux_id_step_{step}': ID_loss_step})
                    
                    # 平均所有步的 ID 损失
                    ID_loss = torch.mean(torch.stack(ID_Losses))
                    loss_aux_total += self.aux_id_loss_weight * ID_loss
                    
                    loss_dict.update({f'{prefix}/loss_aux_id': ID_loss})
                    loss_dict.update({f'{prefix}/aux_sim_imp': sim_imp})
                
                # Step 5: 计算感知损失（LPIPS）
                if self.aux_lpips_loss_weight > 0 and GT_w is not None and hasattr(self, 'lpips_loss'):
                    loss_lpips = 0
                    # 对每步的重构结果和每个尺度都计算 LPIPS
                    for step, x_pred in enumerate(decoded_pred_x_0):
                        for scale_idx in range(3):
                            scale_size = 512 // (2 ** scale_idx)
                            loss_lpips_scale = self.lpips_loss(
                                F.adaptive_avg_pool2d(x_pred, (scale_size, scale_size)),
                                F.adaptive_avg_pool2d(GT_w, (scale_size, scale_size))
                            )
                            loss_lpips += loss_lpips_scale
                            loss_dict.update({
                                f'{prefix}/loss_aux_lpips_step_{step}_scale_{scale_idx}': loss_lpips_scale
                            })
                    
                    loss_aux_total += self.aux_lpips_loss_weight * loss_lpips
                    loss_dict.update({f'{prefix}/loss_aux_lpips': loss_lpips})
            
            # 记录总辅助损失
            if loss_aux_total > 0:
                loss_dict.update({f'{prefix}/loss_aux_total': loss_aux_total})
        
        # ========== 总损失 ==========
        loss = self.dpo_loss_weight * loss_dpo + loss_aux_total
        loss_dict.update({f'{prefix}/loss': loss})
        
        # ⭐ 修复阻塞问题：commit=False，避免每个batch都等待上传
        if wandb.run is not None:
            wandb.log(loss_dict, commit=False)
        return loss, loss_dict


    def p_mean_variance(self, x, c, t, clip_denoised: bool, return_codebook_ids=False, quantize_denoised=False,
                        return_x0=False, score_corrector=None, corrector_kwargs=None):
        t_in = t
        model_out = self.apply_model(x, t_in, c, return_ids=return_codebook_ids)

        if score_corrector is not None:
            assert self.parameterization == "eps"
            model_out = score_corrector.modify_score(self, model_out, x, t, c, **corrector_kwargs)

        if return_codebook_ids:
            model_out, logits = model_out

        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()

        if clip_denoised:
            x_recon.clamp_(-1., 1.)
        if quantize_denoised:
            x_recon, _, [_, _, indices] = self.first_stage_model.quantize(x_recon)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        if return_codebook_ids:
            return model_mean, posterior_variance, posterior_log_variance, logits
        elif return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, c, t, clip_denoised=False, repeat_noise=False,
                 return_codebook_ids=False, quantize_denoised=False, return_x0=False,
                 temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None):
        b, *_, device = *x.shape, x.device
        outputs = self.p_mean_variance(x=x, c=c, t=t, clip_denoised=clip_denoised,
                                       return_codebook_ids=return_codebook_ids,
                                       quantize_denoised=quantize_denoised,
                                       return_x0=return_x0,
                                       score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
        if return_codebook_ids:
            raise DeprecationWarning("Support dropped.")
            model_mean, _, model_log_variance, logits = outputs
        elif return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs

        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        if return_codebook_ids:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, logits.argmax(dim=1)
        if return_x0:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def progressive_denoising(self, cond, shape, verbose=True, callback=None, quantize_denoised=False,
                              img_callback=None, mask=None, x0=None, temperature=1., noise_dropout=0.,
                              score_corrector=None, corrector_kwargs=None, batch_size=None, x_T=None, start_T=None,
                              log_every_t=None):
        if not log_every_t:
            log_every_t = self.log_every_t
        timesteps = self.num_timesteps
        if batch_size is not None:
            b = batch_size if batch_size is not None else shape[0]
            shape = [batch_size] + list(shape)
        else:
            b = batch_size = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=self.device)
        else:
            img = x_T
        intermediates = []
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = tqdm(reversed(range(0, timesteps)), desc='Progressive Generation',
                        total=timesteps) if verbose else reversed(
            range(0, timesteps))
        if type(temperature) == float:
            temperature = [temperature] * timesteps

        for i in iterator:
            ts = torch.full((b,), i, device=self.device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            img, x0_partial = self.p_sample(img, cond, ts,
                                            clip_denoised=self.clip_denoised,
                                            quantize_denoised=quantize_denoised, return_x0=True,
                                            temperature=temperature[i], noise_dropout=noise_dropout,
                                            score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
            if mask is not None:
                assert x0 is not None
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(x0_partial)
            if callback: callback(i)
            if img_callback: img_callback(img, i)
        return img, intermediates

    @torch.no_grad()
    def p_sample_loop(self, cond, shape, return_intermediates=False,
                      x_T=None, verbose=True, callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, start_T=None,
                      log_every_t=None):

        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        intermediates = [img]
        if timesteps is None:
            timesteps = self.num_timesteps

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = tqdm(reversed(range(0, timesteps)), desc='Sampling t', total=timesteps) if verbose else reversed(
            range(0, timesteps))

        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]  # spatial size has to match

        for i in iterator:
            ts = torch.full((b,), i, device=device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            img = self.p_sample(img, cond, ts,
                                clip_denoised=self.clip_denoised,
                                quantize_denoised=quantize_denoised)
            if mask is not None:
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
            if callback: callback(i)
            if img_callback: img_callback(img, i)

        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, cond, batch_size=16, return_intermediates=False, x_T=None,
               verbose=True, timesteps=None, quantize_denoised=False,
               mask=None, x0=None, shape=None,**kwargs):
        if shape is None:
            shape = (batch_size, self.channels, self.image_size, self.image_size)
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]
        return self.p_sample_loop(cond,
                                  shape,
                                  return_intermediates=return_intermediates, x_T=x_T,
                                  verbose=verbose, timesteps=timesteps, quantize_denoised=quantize_denoised,
                                  mask=mask, x0=x0)

    @torch.no_grad()
    def sample_log(self,cond,batch_size,ddim, ddim_steps,**kwargs):

        if ddim:
            ddim_sampler = DDIMSampler(self)
            shape = (self.channels, self.image_size, self.image_size)
            samples, intermediates =ddim_sampler.sample(ddim_steps,batch_size,
                                                        shape,cond,verbose=False,**kwargs)

        else:
            samples, intermediates = self.sample(cond=cond, batch_size=batch_size,
                                                 return_intermediates=True,**kwargs)

        return samples, intermediates
    
    def sample_initial(self,cond,batch_size,ddim, ddim_steps,**kwargs):
        if ddim:
            ddim_sampler = DDIMSampler(self)
            shape = (self.channels, self.image_size, self.image_size)
            samples, intermediates =ddim_sampler.sample(ddim_steps,batch_size,
                                                        shape,cond,verbose=False,**kwargs)

        else:
            samples, intermediates = self.sample(cond=cond, batch_size=batch_size,
                                                 return_intermediates=True,**kwargs)

        return samples, intermediates
    
    @rank_zero_only
    @torch.no_grad()
    def run_test_during_training(self):
        """训练过程中进行测试：生成样本并计算指标"""
        if not hasattr(self, 'test_during_training') or not self.test_during_training:
            return
        
        print(f"\n{'='*60}")
        print(f"[Test During Training] Step {self.global_step}: Starting test evaluation")
        print(f"{'='*60}\n")
        
        try:
            # 1. 从配置中获取测试数据集配置（参考inference_test_bench.py）
            if not hasattr(self.trainer, 'datamodule'):
                print("[Test] No datamodule available, skipping test")
                return
            
            # 从datamodule获取配置
            test_dataloader = None
            try:
                # 尝试从datamodule的dataset_configs获取测试配置
                if hasattr(self.trainer.datamodule, 'dataset_configs') and 'test' in self.trainer.datamodule.dataset_configs:
                    test_config = self.trainer.datamodule.dataset_configs['test']
                    test_args = test_config.get('params', {})
                    
                    # 优先使用self.test_dataset_name参数（参考inference_test_bench.py的opt.dataset逻辑）
                    # 如果未设置，则从配置文件的target中提取
                    dataset_name = getattr(self, 'test_dataset_name', None)
                    if dataset_name is None or dataset_name == "" or dataset_name == "1":
                        # 从配置文件target中提取数据集名称
                        config_target = test_config.get('target', '').split('.')[-1]  # 例如: CelebAdataset
                        if 'CelebAdataset' in config_target:
                            dataset_name = 'CelebA'
                        elif 'FFHQdataset' in config_target:
                            dataset_name = 'FFHQ'
                        elif 'FFdataset' in config_target or 'FF++' in config_target:
                            dataset_name = 'FF++'
                        else:
                            dataset_name = 'CelebA'  # 默认使用CelebA
                    
                    # 创建测试数据集（参考inference_test_bench.py第394-402行）
                    from ldm.data.test_bench_dataset import CelebAdataset, FFHQdataset, FFdataset
                    
                    if dataset_name == 'CelebA':
                        test_dataset = CelebAdataset(split='test', **test_args)
                    elif dataset_name == 'FFHQ':
                        test_dataset = FFHQdataset(split='test', **test_args)
                    elif dataset_name == 'FF++':
                        test_dataset = FFdataset(split='test', **test_args)
                    else:
                        print(f"[Test] Unknown test dataset name: {dataset_name}, using FFHQ as default")
                        test_dataset = FFHQdataset(split='test', **test_args)
                    
                    if test_dataset is not None:
                        # ⭐ 限制数据集大小为test_num_samples，避免加载多余数据
                        import torch.utils.data
                        num_samples = getattr(self, 'test_num_samples', 200)
                        original_size = len(test_dataset)
                        if original_size > num_samples:
                            test_dataset = torch.utils.data.Subset(test_dataset, range(num_samples))
                            print(f"[Test] Limited dataset to {num_samples} samples (original: {original_size} samples)")
                        else:
                            print(f"[Test] Using full test dataset: {original_size} samples")
                        
                        # 创建测试dataloader
                        test_dataloader = torch.utils.data.DataLoader(
                            test_dataset,
                            batch_size=self.test_batch_size,
                            num_workers=4,
                            pin_memory=True,
                            shuffle=False,
                            drop_last=False
                        )
                        print(f"[Test] Created test dataset: {dataset_name}")
                
            except Exception as e:
                print(f"[Test] Failed to create test dataset: {e}")
            
            # 如果没有成功创建test_dataloader，使用validation数据集
            if test_dataloader is None:
                print("[Test] Using validation dataset instead")
                try:
                    test_dataloader = self.trainer.datamodule.val_dataloader()
                    print(f"[Test] Using validation dataloader")
                except Exception as e:
                    print(f"[Test] Failed to get validation dataloader: {e}, skipping test")
                    return
            
            # 3. 创建临时输出目录（在--logdir/tmp下）
            # 获取logdir（优先使用trainer.logdir，这是--logdir参数指定的目录）
            if hasattr(self.trainer, 'logdir'):
                logdir = self.trainer.logdir
            elif hasattr(self.trainer, 'logger') and hasattr(self.trainer.logger, 'log_dir'):
                # 如果没有logdir，回退到logger.log_dir
                logdir = self.trainer.logger.log_dir
            else:
                logdir = "logs"
            
            # 创建tmp目录
            tmp_dir = os.path.join(logdir, "tmp")
            os.makedirs(tmp_dir, exist_ok=True)
            
            # 测试样本保存在 tmp/test_samples_step_{step}/
            test_output_dir = os.path.join(tmp_dir, f"test_samples_step_{self.global_step}")
            os.makedirs(test_output_dir, exist_ok=True)
            
            # 4. 生成样本
            num_samples = getattr(self, 'test_num_samples', 200)
            print(f"[Test] Generating {num_samples} samples...")
            
            samples_generated = 0
            
            with self.ema_scope("Testing"):
                for batch_idx, batch_data in enumerate(test_dataloader):
                    if samples_generated >= num_samples:
                        break
                    
                    # 测试数据集的格式: (test_batch, prior, test_model_kwargs, segment_id_batch)
                    # 参考inference_test_bench.py第441行
                    if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 3:
                        test_batch = batch_data[0].to(self.device)  # [B, 3, H, W] - 目标图像
                        prior = batch_data[1].to(self.device) if len(batch_data) > 1 and isinstance(batch_data[1], torch.Tensor) else test_batch
                        test_model_kwargs = batch_data[2] if isinstance(batch_data[2], dict) else {}
                        segment_id_batch = batch_data[3] if len(batch_data) > 3 else [f"{samples_generated + i:06d}" for i in range(test_batch.shape[0])]
                        
                        # 从test_model_kwargs中提取数据
                        inpaint_image = test_model_kwargs.get('inpaint_image', test_batch)
                        if isinstance(inpaint_image, torch.Tensor):
                            inpaint_image = inpaint_image.to(self.device)
                        else:
                            inpaint_image = test_batch
                            
                        inpaint_mask = test_model_kwargs.get('inpaint_mask', torch.ones_like(test_batch[:, :1]))
                        if isinstance(inpaint_mask, torch.Tensor):
                            inpaint_mask = inpaint_mask.to(self.device)
                        else:
                            inpaint_mask = torch.ones_like(test_batch[:, :1])
                            
                        ref_imgs = test_model_kwargs.get('ref_imgs', test_batch)
                        if isinstance(ref_imgs, torch.Tensor):
                            ref_imgs = ref_imgs.to(self.device)
                            # ref_imgs可能是[B, 1, 224, 224]，需要squeeze
                            if ref_imgs.dim() == 4 and ref_imgs.shape[1] == 1:
                                ref_imgs = ref_imgs.squeeze(1)  # [B, 224, 224]
                                # 如果是单通道，需要转换为RGB
                                if ref_imgs.dim() == 3:
                                    ref_imgs = ref_imgs.unsqueeze(1).repeat(1, 3, 1, 1)  # [B, 3, 224, 224]
                        else:
                            ref_imgs = test_batch
                        
                        # 转换为训练格式的batch
                        batch = {
                            'GT': test_batch,  # 目标图像
                            'inpaint_image': inpaint_image,
                            'inpaint_mask': inpaint_mask,
                            'ref_imgs': ref_imgs,
                            # 测试数据集没有GT_w和GT_l，使用GT代替（这些在测试时不会被使用）
                            'GT_w': test_batch,
                            'GT_l': test_batch,
                        }
                    elif isinstance(batch_data, dict):
                        # 已经是训练格式
                        batch = batch_data
                        for key in batch:
                            if isinstance(batch[key], torch.Tensor):
                                batch[key] = batch[key].to(self.device)
                    else:
                        print(f"[Test] Unknown batch format, skipping")
                        continue
                    
                    batch_size = batch['GT'].shape[0]
                    remaining = num_samples - samples_generated
                    if remaining < batch_size:
                        # 只处理需要的数量
                        for key in batch:
                            if isinstance(batch[key], torch.Tensor):
                                batch[key] = batch[key][:remaining]
                        batch_size = remaining
                    
                    # 获取输入（参考inference_test_bench.py的处理方式）
                    test_batch = batch['GT'].to(self.device)
                    test_model_kwargs = {
                        'inpaint_image': batch['inpaint_image'].to(self.device),
                        'inpaint_mask': batch['inpaint_mask'].to(self.device),
                        'ref_imgs': batch['ref_imgs'].to(self.device)
                    }
                    
                    # 处理 unconditional_conditioning (uc)（参考inference_test_bench.py第498-512行）
                    # 获取测试时的scale参数（默认与推理时保持一致，scale=3.0）
                    test_scale = getattr(self, 'test_guidance_scale', 3.0)
                    uc = None
                    if test_scale != 1.0:
                        uc = self.learnable_vector.repeat(test_batch.shape[0], 1, 1)
                        if hasattr(self, 'stack_feat') and self.stack_feat:
                            uc2 = self.other_learnable_vector.repeat(test_batch.shape[0], 1, 1)
                            uc = torch.cat([uc, uc2], dim=-1)
                    
                    # 获取条件（参考inference_test_bench.py第506-518行）
                    landmarks = self.get_landmarks(test_batch) if (self.Landmark_cond or self.Landmark_loss_weight > 0) else None
                    
                    # 处理ref_imgs的维度
                    ref_imgs_for_cond = test_model_kwargs['ref_imgs']
                    if ref_imgs_for_cond.dim() == 5:  # [B, 1, 3, H, W]
                        ref_imgs_for_cond = ref_imgs_for_cond.squeeze(1)  # [B, 3, H, W]
                    elif ref_imgs_for_cond.dim() == 4 and ref_imgs_for_cond.shape[1] == 1:  # [B, 1, H, W]
                        ref_imgs_for_cond = ref_imgs_for_cond.squeeze(1)  # [B, H, W]
                        if ref_imgs_for_cond.dim() == 3:
                            ref_imgs_for_cond = ref_imgs_for_cond.unsqueeze(1).repeat(1, 3, 1, 1)  # [B, 3, H, W]
                    
                    c = self.conditioning_with_feat(
                        ref_imgs_for_cond.to(torch.float32),
                        landmarks=landmarks,
                        tar=test_batch.to(torch.float32)
                    ).float()
                    
                    if c.shape[-1] == 1024 and hasattr(self, 'proj_out'):
                        c = self.proj_out(c)
                    if len(c.shape) == 2:
                        c = c.unsqueeze(1)
                    
                    # 处理landmarks和uc（如果需要，参考inference_test_bench.py第508-512行）
                    if (hasattr(self, 'land_mark_id_seperate_layers') and self.land_mark_id_seperate_layers) or \
                       (hasattr(self, 'sep_head_att') and self.sep_head_att):
                        if test_scale != 1.0 and landmarks is not None:
                            landmarks_expanded = landmarks.unsqueeze(1) if len(landmarks.shape) != 3 else landmarks
                            uc = torch.cat([uc, landmarks_expanded], dim=-1)
                    
                    # 编码inpaint_image和mask（参考inference_test_bench.py第519-524行）
                    inpaint_image = test_model_kwargs['inpaint_image']
                    inpaint_mask = test_model_kwargs['inpaint_mask']
                    z_inpaint = self.encode_first_stage(inpaint_image)
                    z_inpaint = self.get_first_stage_encoding(z_inpaint).detach()
                    test_model_kwargs['inpaint_image'] = z_inpaint
                    test_model_kwargs['inpaint_mask'] = Resize([z_inpaint.shape[-1], z_inpaint.shape[-1]])(inpaint_mask)
                    
                    # 生成样本（参考inference_test_bench.py第528-538行）
                    from ldm.models.diffusion.ddim import DDIMSampler
                    sampler = DDIMSampler(self)
                    
                    shape = [self.channels, self.image_size, self.image_size]
                    start_code = None  # 不使用固定起始噪声
                    
                    samples_ddim, intermediates = sampler.sample(
                        S=self.test_ddim_steps,  # ddim_steps（参考inference_test_bench.sh，默认50）
                        conditioning=c,
                        batch_size=batch_size,
                        shape=shape,
                        verbose=False,
                        unconditional_guidance_scale=test_scale,  # 使用测试时的scale参数（参考inference_test_bench.sh，默认3.0）
                        unconditional_conditioning=uc,
                        eta=0.0,  # ddim_eta
                        x_T=start_code,
                        log_every_t=100,
                        test_model_kwargs=test_model_kwargs,
                        src_im=ref_imgs_for_cond.to(torch.float32),
                        tar=test_batch.to(torch.float32)
                    )
                    
                    samples = samples_ddim
                    
                    # 解码并保存（参考inference_test_bench.py）
                    x_samples = self.decode_first_stage(samples)
                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                    
                    for i in range(batch_size):
                        img = x_samples[i].cpu().permute(1, 2, 0).numpy()
                        img = (img * 255).astype(np.uint8)
                        # 使用segment_id_batch作为文件名（参考inference_test_bench.py）
                        if 'segment_id_batch' in locals() and isinstance(segment_id_batch, (list, tuple)) and i < len(segment_id_batch):
                            filename = f"{segment_id_batch[i]}.png"
                        else:
                            filename = f"{samples_generated + i:06d}.png"
                        Image.fromarray(img).save(
                            os.path.join(test_output_dir, filename)
                        )
                    
                    samples_generated += batch_size
                    # ⭐ 添加进度输出
                    print(f"[Test] Progress: {samples_generated}/{num_samples} samples generated", end='\r')
                    
                    if samples_generated >= num_samples:
                        print()  # 换行
                        break
            
            print(f"\n[Test] ✅ Completed! Generated {samples_generated} samples, saved to {test_output_dir}")
            
            # 5. 调用评估脚本计算指标
            #try,防止显存不够
            
            self.calculate_test_metrics(test_output_dir, tmp_dir)
            
        except Exception as e:
            print(f"[Test] Error during testing: {e}")
            import traceback
            traceback.print_exc()
    
    @rank_zero_only
    def calculate_test_metrics(self, test_output_dir, tmp_dir):
        """调用评估脚本计算指标"""
        import subprocess
        import re
        
        # 获取测试路径配置
        dataset_name = getattr(self, 'test_dataset_name', 'celeba')
        # 将数据集名称映射为ID_retrieval.py期望的格式（参考ID_retrieval.py第178-185行）
        # ID_retrieval.py支持: "celeba", "ffhq", "ff++"
        dataset_name_lower = dataset_name.lower() if dataset_name else 'celeba'
        if dataset_name_lower == 'ffhq':
            id_retrieval_dataset = 'ffhq'
        elif dataset_name_lower == 'ff++':
            id_retrieval_dataset = 'ff++'
        else:
            id_retrieval_dataset = 'celeba'  # 默认使用celeba
        
        source_path = getattr(self, 'test_source_path', None)
        target_path = getattr(self, 'test_target_path', None)
        source_mask_path = getattr(self, 'test_source_mask_path', None)
        target_mask_path = getattr(self, 'test_target_mask_path', None)
        
        if not target_path:
            print("[Test] test_target_path not configured, skipping metrics calculation")
            return
        
        # 设置日志文件（保存在tmp目录下）
        metrics_log_file = os.path.join(tmp_dir, f"test_metrics_step_{self.global_step}.txt")
        
        print(f"\n[Test] Calculating metrics, results will be saved to {metrics_log_file}")
        
        # 获取项目根目录
        import pathlib
        project_root = pathlib.Path(__file__).parent.parent.parent.parent
        project_root_str = str(project_root)
        
        # 获取当前CUDA设备
        if torch.cuda.is_available():
            cuda_device = str(self.device.index if hasattr(self.device, 'index') and self.device.index is not None else 0)
        else:
            cuda_device = None
        
        metrics_results = {}
        
        # 构建基础命令前缀（参考ID_retrieval.py的命令格式）
        def build_cmd_prefix(script_path):
            """构建包含CUDA_VISIBLE_DEVICES和PYTHONPATH的命令前缀"""
            cmd_parts = []
            if cuda_device is not None:
                cmd_parts.append(f"CUDA_VISIBLE_DEVICES={cuda_device}")
            cmd_parts.append(f'PYTHONPATH="{project_root_str}:$PYTHONPATH"')
            cmd_parts.append(f"python {script_path}")
            return " \\\n    ".join(cmd_parts)
        
        # 1. 计算 Pose 距离
        if target_path:
            print("[Test] Calculating Pose distance...")
            try:
                pose_output_file = os.path.join(tmp_dir, f'test_pose_step_{self.global_step}.txt')
                script_path = str(project_root / 'eval_tool' / 'Pose' / 'pose_compare.py')
                cmd_prefix = build_cmd_prefix(script_path)
                cmd_args = f"{target_path} {test_output_dir} --max-samples {getattr(self, 'test_num_samples', 200)} --output {pose_output_file} --device {'cuda' if torch.cuda.is_available() else 'cpu'}"
                cmd = f"{cmd_prefix} \\\n    {cmd_args}"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=project_root_str)
                if result.returncode == 0:
                    # 从输出文件读取结果
                    if os.path.exists(pose_output_file):
                        with open(pose_output_file, 'r') as f:
                            content = f.read()
                            match = re.search(r'Pose_value:\s*([\d.]+)', content)
                            if match:
                                pose_value = float(match.group(1))
                                metrics_results['pose_distance'] = pose_value
                                self.log('test/pose_distance', pose_value, on_step=True, prog_bar=True)
                                print(f"[Test] Pose Distance: {pose_value:.4f}")
                else:
                    print(f"[Test] Pose calculation failed with return code {result.returncode}")
                    print(f"[Test] Error: {result.stderr}")
            except Exception as e:
                print(f"[Test] Pose calculation failed: {e}")
        
        # 2. 计算 Expression 距离
        if target_path:
            print("[Test] Calculating Expression distance...")
            try:
                expr_output_file = os.path.join(tmp_dir, f'test_expression_step_{self.global_step}.txt')
                script_path = str(project_root / 'eval_tool' / 'Expression' / 'expression_compare_face_recon.py')
                cmd_prefix = build_cmd_prefix(script_path)
                cmd_args = f"{target_path} {test_output_dir} --max-samples {getattr(self, 'test_num_samples', 200)} --output {expr_output_file} --device {'cuda' if torch.cuda.is_available() else 'cpu'}"
                cmd = f"{cmd_prefix} \\\n    {cmd_args}"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=project_root_str)
                if result.returncode == 0:
                    # 从输出文件读取结果
                    if os.path.exists(expr_output_file):
                        with open(expr_output_file, 'r') as f:
                            content = f.read()
                            match = re.search(r'Expression_value:\s*([\d.]+)', content)
                            if match:
                                expr_value = float(match.group(1))
                                metrics_results['expression_distance'] = expr_value
                                self.log('test/expression_distance', expr_value, on_step=True, prog_bar=True)
                                print(f"[Test] Expression Distance: {expr_value:.4f}")
                else:
                    print(f"[Test] Expression calculation failed with return code {result.returncode}")
                    print(f"[Test] Error: {result.stderr}")
            except Exception as e:
                print(f"[Test] Expression calculation failed: {e}")
        
        # 3. 计算 ID Retrieval（如果有mask路径）
        if source_path and source_mask_path and target_mask_path:
            print("[Test] Calculating ID Retrieval...")
            try:
                id_output_file = os.path.join(tmp_dir, f'test_id_step_{self.global_step}.txt')
                script_path = str(project_root / 'eval_tool' / 'ID_retrieval' / 'ID_retrieval.py')
                cmd_prefix = build_cmd_prefix(script_path)
                cmd_args = f"{source_path} {test_output_dir} {source_mask_path} {target_mask_path} --max-samples {getattr(self, 'test_num_samples', 200)} --output {id_output_file} --arcface True --dataset {id_retrieval_dataset}"
                cmd = f"{cmd_prefix} \\\n    {cmd_args}"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=project_root_str)
                if result.returncode == 0:
                    # 从输出文件读取结果
                    if os.path.exists(id_output_file):
                        with open(id_output_file, 'r') as f:
                            content = f.read()
                            top1_match = re.search(r'Top-1 accuracy:\s*([\d.]+)%', content)
                            top5_match = re.search(r'Top-5 accuracy:\s*([\d.]+)%', content)
                            mean_match = re.search(r'Mean ID feat:\s*([\d.]+)', content)
                            if top1_match:
                                top1 = float(top1_match.group(1)) / 100.0
                                metrics_results['id_top1'] = top1
                                self.log('test/id_top1', top1, on_step=True, prog_bar=True)
                                print(f"[Test] ID Top-1: {top1*100:.2f}%")
                            if top5_match:
                                top5 = float(top5_match.group(1)) / 100.0
                                metrics_results['id_top5'] = top5
                                self.log('test/id_top5', top5, on_step=True, prog_bar=True)
                                print(f"[Test] ID Top-5: {top5*100:.2f}%")
                            if mean_match:
                                mean_id = float(mean_match.group(1))
                                metrics_results['id_mean'] = mean_id
                                self.log('test/id_mean', mean_id, on_step=True, prog_bar=True)
                                print(f"[Test] ID Mean: {mean_id:.4f}")
                else:
                    print(f"[Test] ID Retrieval calculation failed with return code {result.returncode}")
                    print(f"[Test] Error: {result.stderr}")
            except Exception as e:
                print(f"[Test] ID Retrieval calculation failed: {e}")
        
        # 4. 计算 FID（如果有数据集路径）
        dataset_path = getattr(self, 'test_dataset_path', None)
        if dataset_path and os.path.exists(dataset_path):
            print("[Test] Calculating FID score...")
            try:
                fid_output_file = os.path.join(tmp_dir, f'test_fid_step_{self.global_step}.txt')
                script_path = str(project_root / 'eval_tool' / 'fid' / 'fid_score.py')
                cmd_prefix = build_cmd_prefix(script_path)
                cmd_args = f"{dataset_path} {test_output_dir} --max-samples {getattr(self, 'test_num_samples', 200)} --output {fid_output_file} --device {'cuda' if torch.cuda.is_available() else 'cpu'}"
                cmd = f"{cmd_prefix} \\\n    {cmd_args}"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=project_root_str)
                if result.returncode == 0:
                    # 从输出文件读取结果
                    if os.path.exists(fid_output_file):
                        with open(fid_output_file, 'r') as f:
                            content = f.read()
                            # 匹配 "FID Score:" 或 "FID score:"（大小写不敏感）
                            match = re.search(r'FID\s+[Ss]core:\s*([\d.]+)', content, re.IGNORECASE)
                            if match:
                                fid_value = float(match.group(1))
                                metrics_results['fid'] = fid_value
                                self.log('test/fid', fid_value, on_step=True, prog_bar=True)
                                print(f"[Test] FID Score: {fid_value:.4f}")
                            else:
                                print(f"[Test] Failed to parse FID score from output file")
                                print(f"[Test] File content preview: {content[:200]}")
                else:
                    print(f"[Test] FID calculation failed with return code {result.returncode}")
                    print(f"[Test] Error: {result.stderr}")
            except Exception as e:
                print(f"[Test] FID calculation failed: {e}")
        
        # 5. 保存汇总结果到日志文件
        with open(metrics_log_file, 'w', encoding='utf-8') as f:
            f.write(f"Test Metrics at Step {self.global_step}\n")
            f.write("="*60 + "\n\n")
            for key, value in metrics_results.items():
                f.write(f"{key}: {value:.6f}\n")
            f.write("\n" + "="*60 + "\n")
        
        # 6. 记录到 wandb（类似 log_images 的方式）
        self.log_test_metrics_to_wandb(metrics_results)
        
        print(f"\n[Test] All metrics calculated. Summary saved to {metrics_log_file}")
        print(f"[Test] Metrics: {metrics_results}\n")
    
    def log_test_metrics_to_wandb(self, metrics_results):
        """将测试指标记录到 wandb，类似 log_images 的方式"""
        if not metrics_results:
            return
        
        try:
            # 方法1: 使用 wandb.Table 创建表格（推荐，更直观）
            import wandb
            
            if wandb.run is not None:
                # 创建指标表格
                table = wandb.Table(columns=["Metric", "Value", "Step"])
                for metric_name, metric_value in metrics_results.items():
                    table.add_data(metric_name, f"{metric_value:.6f}", self.global_step)
                
                # 记录表格到 wandb
                wandb.log({
                    "test/metrics_table": table,
                    "global_step": self.global_step
                }, commit=False)
                
                # 方法2: 同时记录为单独的指标（用于绘制曲线图）
                # 这些已经通过 self.log() 记录了，这里再记录一次确保同步
                for metric_name, metric_value in metrics_results.items():
                    wandb.log({
                        f"test/{metric_name}": metric_value,
                        "global_step": self.global_step
                    }, commit=False)
                
                # 方法3: 记录汇总信息到 summary（可选）
                # 只在特定步骤更新 summary（例如每1000步）
                if self.global_step % 1000 == 0:
                    summary_dict = {f"test_summary/{k}": v for k, v in metrics_results.items()}
                    wandb.run.summary.update(summary_dict)
                
                # ⭐ 修复阻塞问题：不再强制提交，让wandb后台异步处理
                # wandb.log({}, commit=True)  # 移除阻塞的提交
                
                # 打印详细的成功信息
                print(f"[Test] ✅ Successfully logged {len(metrics_results)} metrics to wandb:")
                for metric_name, metric_value in metrics_results.items():
                    print(f"  - test/{metric_name}: {metric_value:.6f}")
                print(f"[Test] Check wandb dashboard for:")
                print(f"  - Table: test/metrics_table")
                print(f"  - Plots: test/pose_distance, test/expression_distance, test/id_top1, test/id_top5, test/id_mean, test/fid")
        except Exception as e:
            print(f"[Test] ❌ Failed to log metrics to wandb: {e}")
            import traceback
            traceback.print_exc()

    @torch.no_grad()
    def log_images(self, batch, N=4, sample=True, ddim_steps=50, ddim_eta=1., return_keys=None, **kwargs):
        """
        记录SFT或DPO训练的关键图片
        
        Args:
            batch: 数据批次
            N: 记录的样本数量
            sample: 是否生成模型输出
            ddim_steps: DDIM采样步数
            ddim_eta: DDIM eta参数
            
        Returns:
            log: 包含以下内容的字典
                - ref_imgs: 参考人脸（已mask）
                - tgt: 目标图像
                - tgt_mask: 目标mask
                - x_w: 赢样本（好的换脸结果）
                - output_current: 当前训练模型（EMA）生成的输出（可选）
                - output_reference: 参考模型生成的输出（可选，仅DPO模式）
        """
        mode = "DPO" if self.model_ref is not None else "SFT"
        print(f"[{mode} log_images] Called at step {self.global_step}, N={N}, sample={sample}")
        
        try:
            use_ddim = ddim_steps is not None
            log = dict()
            
            print(f"[{mode} log_images] Getting input data...")
            z, _, c, x_w, x_l, xrec_w, xrec_l, reference, mask, _, gt, inpaint_src = self.get_input(
                batch, self.first_stage_key,
                return_first_stage_outputs=True,
                force_c_encode=True,
                return_original_cond=True,
                get_mask=True,
                get_reference=True,
                get_gt=True,
                get_inpaint=True,
                bs=N
            )

            N = min(x_w.shape[0], N)
            
            # 收集所有需要拼接的图像
            images_to_concat = []
            target_size = 512  # 统一尺寸
            
            # 1. 参考图像（已mask）
            if reference is not None:
                ref_normalized = un_norm_clip(reference)
                ref_normalized = TF.normalize(ref_normalized, 
                                             mean=[0.5, 0.5, 0.5], 
                                             std=[0.5, 0.5, 0.5])
                ref_resized = TF.resize(ref_normalized, (target_size, target_size))
                images_to_concat.append(ref_resized)
            
            # 2. 目标图像
            tgt_resized = TF.resize(gt, (target_size, target_size))
            images_to_concat.append(tgt_resized)
            
            # 3. mask后的目标图像（直接使用数据集提供的inpaint_src）
            tgt_masked_resized = TF.resize(inpaint_src, (target_size, target_size))
            images_to_concat.append(tgt_masked_resized)
            
            # 4. 赢样本
            x_w_resized = TF.resize(x_w, (target_size, target_size))
            images_to_concat.append(x_w_resized)
            
            print(f"[{mode} log_images] Added 4 base images (ref_imgs, tgt, tgt_mask, x_w)")

            # 生成模型输出
            if sample:
                print(f"[{mode} log_images] Generating model outputs...")
                
                # 5. 使用当前训练模型（EMA）生成
                print(f"[{mode} log_images] Sampling with current model (EMA)...")
                with self.ema_scope("Plotting"):
                    if self.first_stage_key == 'inpaint':
                        samples_current, _ = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
                                                    ddim_steps=ddim_steps, eta=ddim_eta, rest=z[:, 4:, :, :])
                    else:
                        samples_current, _ = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
                                                    ddim_steps=ddim_steps, eta=ddim_eta)
                x_samples_current = self.decode_first_stage(samples_current)
                x_samples_current_resized = TF.resize(x_samples_current, (target_size, target_size))
                images_to_concat.append(x_samples_current_resized)
                print(f"[{mode} log_images] Added current model output")
                
                # 6. 使用参考模型生成（仅在DPO模式下）
                if self.model_ref is not None:
                    print(f"[DPO log_images] Sampling with reference model...")
                    # 临时替换模型为参考模型
                    model_backup = self.model
                    self.model = self.model_ref
                    try:
                        if self.first_stage_key == 'inpaint':
                            samples_ref, _ = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
                                                        ddim_steps=ddim_steps, eta=ddim_eta, rest=z[:, 4:, :, :])
                        else:
                            samples_ref, _ = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
                                                        ddim_steps=ddim_steps, eta=ddim_eta)
                        x_samples_ref = self.decode_first_stage(samples_ref)
                        x_samples_ref_resized = TF.resize(x_samples_ref, (target_size, target_size))
                        images_to_concat.append(x_samples_ref_resized)
                        print(f"[DPO log_images] Added reference model output")
                    finally:
                        # 恢复原模型
                        self.model = model_backup
                else:
                    print(f"[SFT log_images] Skipping reference model (SFT mode)")
                    
            else:
                print(f"[{mode} log_images] Skipping model output (sample={sample})")

            # 竖着拼接所有图像 (每个样本单独拼接)
            concatenated_images = []
            for i in range(N):
                sample_images = [img[i:i+1] for img in images_to_concat]  # 取第i个样本
                # 在高度维度拼接 [1, 3, H, W] -> [1, 3, H*n, W]
                concat_img = torch.cat(sample_images, dim=2)
                concatenated_images.append(concat_img)
            
            # 合并所有样本 [N, 3, H*n, W]
            log["concatenated"] = torch.cat(concatenated_images, dim=0)
            
            print(f"[{mode} log_images] Successfully created concatenated image with {len(images_to_concat)} rows x {N} samples")
            
            if return_keys:
                if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                    return log
                else:
                    return {key: log[key] for key in return_keys}
            return log
            
        except Exception as e:
            print(f"[{mode} log_images] ERROR: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        
        if self.partial_training:
        # if True:
            print("Partial training.............................")
            train_names=self.trainable_keys
            train_names=[ 'attn2','norm2']
            params_train=[]
            for name,param in self.model.named_parameters():
                if "diffusion_model" not in name and param.requires_grad:
                    print(name)
                    params_train.append(param)
                    
                elif "diffusion_model" in name and any(train_name in name for train_name in train_names):
                    print(name)
                    params_train.append(param)
            params=params_train
        print("Setting up Adam optimizer.......................")

        if self.cond_stage_trainable:
            print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            if self.ID_weight>0:
                params = params + list(self.cond_stage_model.final_ln.parameters())+list(self.cond_stage_model.mapper.parameters())+list(self.proj_out.parameters())+list(self.ID_proj_out.parameters())  # changed
            else:
                params = params + list(self.cond_stage_model.final_ln.parameters())+list(self.cond_stage_model.mapper.parameters())+list(self.proj_out.parameters())
        if self.learn_logvar:
            print('Diffusion model optimizing logvar')
            params.append(self.logvar)
        params.append(self.learnable_vector)
        opt = torch.optim.AdamW(params, lr=lr)
        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [opt], scheduler
        return opt

    @torch.no_grad()
    def to_rgb(self, x):
        x = x.float()
        if not hasattr(self, "colorize"):
            self.colorize = torch.randn(3, x.shape[1], 1, 1).to(x)
        x = nn.functional.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x


class DiffusionWrapper(pl.LightningModule):
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.conditioning_key = conditioning_key
        assert self.conditioning_key in [None, 'concat', 'crossattn', 'hybrid', 'adm']

    def forward(self, x, t, c_concat: list = None, c_crossattn: list = None,return_features=False):
        if self.conditioning_key is None:
            out = self.diffusion_model(x, t)
        elif self.conditioning_key == 'concat':
            xc = torch.cat([x] + c_concat, dim=1)
            out = self.diffusion_model(xc, t)
        elif self.conditioning_key == 'crossattn':
            cc = torch.cat(c_crossattn, 1)  #-->cc.shape = (bs, 1, 768) ## adding return_features  here only for testing
            out = self.diffusion_model(x, t, context=cc,return_features=return_features)
        elif self.conditioning_key == 'hybrid':
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc)
        elif self.conditioning_key == 'adm':
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, y=cc)
        else:
            raise NotImplementedError()

        return out  #-->out.shape = (bs, 4,64,64)


class Layout2ImgDiffusion(LatentDiffusion):
    # TODO: move all layout-specific hacks to this class
    def __init__(self, cond_stage_key, *args, **kwargs):
        assert cond_stage_key == 'coordinates_bbox', 'Layout2ImgDiffusion only for cond_stage_key="coordinates_bbox"'
        super().__init__(cond_stage_key=cond_stage_key, *args, **kwargs)

    def log_images(self, batch, N=8, *args, **kwargs):
        logs = super().log_images(batch=batch, N=N, *args, **kwargs)

        key = 'train' if self.training else 'validation'
        dset = self.trainer.datamodule.datasets[key]
        mapper = dset.conditional_builders[self.cond_stage_key]

        bbox_imgs = []
        map_fn = lambda catno: dset.get_textual_label(dset.get_category_id(catno))
        for tknzd_bbox in batch[self.cond_stage_key][:N]:
            bboximg = mapper.plot(tknzd_bbox.detach().cpu(), map_fn, (256, 256))
            bbox_imgs.append(bboximg)

        cond_img = torch.stack(bbox_imgs, dim=0)
        logs['bbox_image'] = cond_img
        return logs

class LatentInpaintDiffusion(LatentDiffusion):
    def __init__(
        self,
        concat_keys=("mask", "masked_image"),
        masked_image_key="masked_image",
        finetune_keys=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.masked_image_key = masked_image_key
        assert self.masked_image_key in concat_keys
        self.concat_keys = concat_keys


    @torch.no_grad()
    def get_input(
        self, batch, k, cond_key=None, bs=None, return_first_stage_outputs=False
    ):
        # note: restricted to non-trainable encoders currently
        assert (
            not self.cond_stage_trainable
        ), "trainable cond stages not yet supported for inpainting"
        z, c, x, xrec, xc = super().get_input(
            batch,
            self.first_stage_key,
            return_first_stage_outputs=True,
            force_c_encode=True,
            return_original_cond=True,
            bs=bs,
        )

        assert exists(self.concat_keys)
        c_cat = list()
        for ck in self.concat_keys:
            cc = (
                rearrange(batch[ck], "b h w c -> b c h w")
                .to(memory_format=torch.contiguous_format)
                .float()
            )
            if bs is not None:
                cc = cc[:bs]
                cc = cc.to(self.device)
            bchw = z.shape
            if ck != self.masked_image_key:
                cc = torch.nn.functional.interpolate(cc, size=bchw[-2:])
            else:
                cc = self.get_first_stage_encoding(self.encode_first_stage(cc))
            c_cat.append(cc)
        c_cat = torch.cat(c_cat, dim=1)
        all_conds = {"c_concat": [c_cat], "c_crossattn": [c]}
        if return_first_stage_outputs:
            return z, all_conds, x, xrec, xc
        return z, all_conds
