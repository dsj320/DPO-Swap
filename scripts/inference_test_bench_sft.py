import argparse, os, sys, glob
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
# import pandas as pd
from PIL import Image
from tqdm import tqdm, trange
# from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
import torchvision
from ldm.util import instantiate_from_config

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor

from ldm.data.test_bench_dataset import COCOImageDataset
from ldm.data.test_bench_dataset_sft import CelebAdataset,FFHQdataset,FFdataset
# import clip
from torchvision.transforms import Resize


from PIL import Image
from torchvision.transforms import PILToTensor




# from dift.src.models.dift_sd import SDFeaturizer
# from dift.src.utils.visualization import Demo


# import matplotlib.pyplot as plt
import torch.nn as nn

# cos = nn.CosineSimilarity(dim=0)
import numpy as np  

# load safety model - DISABLED for offline use
safety_model_id = "CompVis/stable-diffusion-safety-checker"
# safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
# safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)
safety_feature_extractor = None
safety_checker = None

# set cuda device 

def save_sample_by_decode(x,model,Base_path,segment_id_batch,intermediate_num):

    
    x = model.decode_first_stage(x)
    x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
    x = x.cpu().permute(0, 2, 3, 1).numpy()
    for i in range(len(x)):
        img = Image.fromarray((x[i] * 255).astype(np.uint8))
        save_path=os.path.join(Base_path, f"{segment_id_batch[i]}")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        img.save(os.path.join(Base_path, f"{segment_id_batch[i]}/{intermediate_num}.png"))
    


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def check_safety(x_image):
    # Safety checker disabled for offline use
    if safety_checker is None or safety_feature_extractor is None:
        return x_image, [False] * len(x_image)
    
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a photograph of an astronaut riding a horse",
        help="the prompt to render"
    )
    parser.add_argument(
        "--device_ID",
        type=int,
        default=5,
        help="device_ID",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="results/debug"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--Guidance",
        action='store_true',
        help="Guidance in inference ",
    )
    parser.add_argument(
        "--Start_from_target",
        action='store_true',
        help="if enabled, uses the noised target image as the starting ",
    )
    parser.add_argument(
        "--target_start_noise_t",
        type=int,
        default=1000,
        help="target_start_noise_t",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=2,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=5,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="dataset: CelebA,FFHQ,FF",
        default='CelebA'
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        help="dataset_dir",
        default='dataset/FaceData/CelebAMask-HQ'
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="models/REFace/configs/project_ffhq.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/REFace/checkpoints/last.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="full"
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="start index for inference (for parallel processing)",
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=None,
        help="end index for inference (None means process all remaining images)",
    )
    opt = parser.parse_args()
    
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    print(opt)
    if opt.laion400m:
        print("Falling back to LAION 400M model...")
        opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        opt.ckpt = "models/ldm/text2img-large/model.ckpt"
        opt.outdir = "outputs/txt2img-samples-laion400m"

    seed_everything(opt.seed)
    # torch.cuda.set_device(opt.device_ID)
    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)


    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir


    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))

    sample_path = os.path.join(outpath, "samples")
    result_path = os.path.join(outpath, "results")
    grid_path=os.path.join(outpath, "grid")
    os.makedirs(sample_path, exist_ok=True)
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(grid_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

 

    # test_dataset=COCOImageDataset(test_bench_dir='test_bench') 
    #read config file :configs/v2.yaml
    conf_file=OmegaConf.load(opt.config)
    # breakpoint()
    
    # 优先尝试从配置文件直接实例化数据集（支持13ch等特殊模型）
    if 'data' in conf_file and 'params' in conf_file.data and 'test' in conf_file.data.params:
        test_config = conf_file.data.params.test
        # 检查是否是 SFTFaceDataset_13ch
        if 'target' in test_config and 'sft_dataset_13ch' in test_config.target:
            from ldm.data.sft_dataset_13ch import SFTFaceDataset
            test_args = test_config.params if 'params' in test_config else {}
            print(f"✅ 使用配置文件中的数据集: {test_config.target}")
            print(f"   - data_manifest_path: {test_args.get('data_manifest_path', 'N/A')}")
            print(f"   - base_3d_path: {test_args.get('base_3d_path', 'N/A')}")
            test_dataset = SFTFaceDataset(**test_args)
        else:
            # 回退到原来的方式
            test_args=conf_file.data.params.test.params
            
            if opt.dataset=='CelebA':
                test_dataset=CelebAdataset(state='test',**test_args)
                
            elif opt.dataset=='FFHQ':
                test_dataset=FFHQdataset(state='test',**test_args)
                
            elif opt.dataset=='FF++':
                test_args['dataset_dir']=opt.dataset_dir if opt.dataset_dir is not None else test_args['dataset_dir']
                test_dataset=FFdataset(state='test',**test_args)
    else:
        raise ValueError("Config file must contain 'data.params.test' section")
        
    test_dataloader= torch.utils.data.DataLoader(test_dataset, 
                                        batch_size=batch_size, 
                                        num_workers=4, 
                                        pin_memory=True, 
                                        shuffle=False,
                                        drop_last=False)

    # 设置推理范围
    total_samples = len(test_dataset)
    start_idx = opt.start_idx
    end_idx = opt.end_idx if opt.end_idx is not None else total_samples
    
    # 验证索引范围
    if start_idx < 0 or start_idx >= total_samples:
        raise ValueError(f"start_idx ({start_idx}) must be in range [0, {total_samples})")
    if end_idx <= start_idx or end_idx > total_samples:
        raise ValueError(f"end_idx ({end_idx}) must be in range ({start_idx}, {total_samples}]")
    
    print(f"\n{'='*80}")
    print(f"🎯 Processing samples {start_idx} to {end_idx-1} (total: {end_idx - start_idx} samples)")
    print(f"📊 Dataset total size: {total_samples}")
    print(f"{'='*80}\n")

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

   
    use_prior=True
    
    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    sample_idx = 0  # 当前处理的全局样本索引
    processed_count = 0  # 实际处理的样本数量
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                all_samples = list()
                for batch_data in test_dataloader:
                    # ⭐ 支持两种数据集格式：字典格式（SFTFaceDataset_13ch）和元组格式（旧数据集）
                    if isinstance(batch_data, dict):
                        # SFTFaceDataset_13ch 返回字典格式
                        batch = batch_data
                        # 转移到设备
                        for key in batch:
                            if isinstance(batch[key], torch.Tensor):
                                batch[key] = batch[key].to(device)
                        
                        # 提取关键数据
                        test_batch = batch['GT']
                        prior = batch.get('GT_nomask', test_batch)
                        test_model_kwargs = {
                            'inpaint_image': batch['inpaint_image'],
                            'inpaint_mask': batch['inpaint_mask'],
                            'ref_imgs': batch['ref_imgs'],
                        }
                        # 13通道模型需要 mixed_3d_landmarks
                        if 'mixed_3d_landmarks' in batch:
                            test_model_kwargs['mixed_3d_landmarks'] = batch['mixed_3d_landmarks']
                        
                        # 生成 segment_id（使用索引）
                        batch_size = test_batch.shape[0]
                        segment_id_batch = [f"{sample_idx + i:06d}" for i in range(batch_size)]
                    
                    elif isinstance(batch_data, (list, tuple)) and len(batch_data) >= 4:
                        # 旧数据集格式：(test_batch, prior, test_model_kwargs, segment_id_batch)
                        test_batch, prior, test_model_kwargs, segment_id_batch = batch_data
                    else:
                        raise ValueError(f"Unsupported batch format: {type(batch_data)}")
                    
                    # 计算当前batch的样本索引范围
                    batch_start_idx = sample_idx
                    batch_end_idx = sample_idx + test_batch.shape[0]
                    
                    # 跳过start_idx之前的batch
                    if batch_end_idx <= start_idx:
                        sample_idx = batch_end_idx
                        continue
                    
                    # 如果已经处理完end_idx，停止
                    if batch_start_idx >= end_idx:
                        break
                    
                    # 处理部分batch的情况（batch跨越start或end边界）
                    process_start = max(0, start_idx - batch_start_idx)
                    process_end = min(test_batch.shape[0], end_idx - batch_start_idx)
                    
                    # 如果需要截取batch
                    if process_start > 0 or process_end < test_batch.shape[0]:
                        test_batch = test_batch[process_start:process_end]
                        prior = prior[process_start:process_end]
                        test_model_kwargs = {k: v[process_start:process_end] if isinstance(v, torch.Tensor) else v 
                                           for k, v in test_model_kwargs.items()}
                        segment_id_batch = segment_id_batch[process_start:process_end]
                    
                    current_global_idx = batch_start_idx + process_start
                    
                    # ✅ 检查batch中哪些样本已经存在
                    existing_mask = []
                    for seg_id in segment_id_batch:
                        result_file = os.path.join(result_path, f"{seg_id}.png")
                        existing_mask.append(os.path.exists(result_file))
                    
                    # 如果所有样本都已存在，跳过整个batch
                    if all(existing_mask):
                        print(f"⏭️  跳过batch {current_global_idx}-{current_global_idx + len(segment_id_batch) - 1} (已全部完成)")
                        sample_idx = batch_end_idx
                        processed_count += len(segment_id_batch)
                        continue
                    
                    # 统计跳过的样本数
                    skipped_count = sum(existing_mask)
                    if skipped_count > 0:
                        print(f"⏭️  Batch中跳过 {skipped_count}/{len(segment_id_batch)} 个已存在样本")
                    
                    processed_count += len(segment_id_batch)
                    print(f"Processing: {processed_count}/{end_idx - start_idx} | Global idx: {current_global_idx}-{current_global_idx + len(segment_id_batch) - 1}      ", end='\r')
                    
                    sample_idx = batch_end_idx
                    
                    if opt.Start_from_target:
                        
                        x=test_batch
                        x=x.to(device)
                        encoder_posterior = model.encode_first_stage(x)
                        z = model.get_first_stage_encoding(encoder_posterior)
                        t=int(opt.target_start_noise_t)
                        # t = torch.ones((x.shape[0],), device=device).long()*t
                        t = torch.randint(t-1, t, (x.shape[0],), device=device).long()
                    
                        if use_prior:
                            prior=prior.to(device)
                            encoder_posterior_2=model.encode_first_stage(prior)
                            z2 = model.get_first_stage_encoding(encoder_posterior_2)
                            noise = torch.randn_like(z2)
                            x_noisy = model.q_sample(x_start=z2, t=t, noise=noise)
                            start_code = x_noisy
                            # print('start from target')
                        else:
                            noise = torch.randn_like(z)
                            x_noisy = model.q_sample(x_start=z, t=t, noise=noise)
                            start_code = x_noisy
                        # print('start from target')
                        
                    test_model_kwargs={n:test_model_kwargs[n].to(device,non_blocking=True) for n in test_model_kwargs }
                    uc = None
                    if opt.scale != 1.0:
                        uc = model.learnable_vector.repeat(test_batch.shape[0],1,1)
                        if model.stack_feat:
                            uc2=model.other_learnable_vector.repeat(test_batch.shape[0],1,1)
                            uc=torch.cat([uc,uc2],dim=-1)
                    
                    # ⭐ 修复1：完整的ref_imgs维度处理（参考训练代码 ddpm_sft_13ch.py:3029-3036）
                    landmarks=model.get_landmarks(test_batch) if model.Landmark_cond else None
                    
                    ref_imgs_for_cond = test_model_kwargs['ref_imgs']
                    if ref_imgs_for_cond.dim() == 5:  # [B, 1, 3, H, W]
                        ref_imgs_for_cond = ref_imgs_for_cond.squeeze(1)  # [B, 3, H, W]
                    elif ref_imgs_for_cond.dim() == 4 and ref_imgs_for_cond.shape[1] == 1:  # [B, 1, H, W]
                        ref_imgs_for_cond = ref_imgs_for_cond.squeeze(1)  # [B, H, W]
                        if ref_imgs_for_cond.dim() == 3:
                            ref_imgs_for_cond = ref_imgs_for_cond.unsqueeze(1).repeat(1, 3, 1, 1)  # [B, 3, H, W]
                    
                    c=model.conditioning_with_feat(ref_imgs_for_cond.to(torch.float32),landmarks=landmarks,tar=test_batch.to(torch.float32)).float()
                    if (model.land_mark_id_seperate_layers or model.sep_head_att) and opt.scale != 1.0:
            
                        # concat c, landmarks
                        landmarks=landmarks.unsqueeze(1) if len(landmarks.shape)!=3 else landmarks
                        uc=torch.cat([uc,landmarks],dim=-1)
                    
                    
                    if c.shape[-1]==1024:
                        c = model.proj_out(c)
                    if len(c.shape)==2:
                        c = c.unsqueeze(1)
                    inpaint_image=test_model_kwargs['inpaint_image']
                    inpaint_mask=test_model_kwargs['inpaint_mask']
                    z_inpaint = model.encode_first_stage(test_model_kwargs['inpaint_image'])
                    z_inpaint = model.get_first_stage_encoding(z_inpaint).detach()
                    test_model_kwargs['inpaint_image']=z_inpaint
                    test_model_kwargs['inpaint_mask']=Resize([z_inpaint.shape[-1],z_inpaint.shape[-1]])(test_model_kwargs['inpaint_mask'])

                    # 对于13通道模型，需要编码 mixed_3d_landmarks（参考 ddpm_sft_13ch.py 第3062-3074行）
                    sampler_kwargs = {}
                    if 'mixed_3d_landmarks' in test_model_kwargs:
                        print("✅ 检测到13通道模型，处理 mixed_3d_landmarks")
                        landmarks_input = test_model_kwargs['mixed_3d_landmarks']
                        # 使用 landmark_encoder 编码（参考训练代码第3068行）
                        landmark_latent = None
                        if hasattr(model, 'landmark_encoder') and landmarks_input is not None:
                            landmark_latent = model.landmark_encoder(landmarks_input)  # [B, 4, 64, 64]
                        # 兜底：如果没有编码器，使用零张量
                        if landmark_latent is None:
                            landmark_latent = torch.zeros_like(z_inpaint)  # [B, 4, 64, 64]
                        
                        # 构造 rest 参数：[z_inpaint, mask, landmark_latent]（参考训练代码第3073行）
                        mask_resize = test_model_kwargs['inpaint_mask']
                        rest = torch.cat([z_inpaint, mask_resize, landmark_latent], dim=1)
                        sampler_kwargs['rest'] = rest
                        print(f"   - rest shape: {rest.shape} (应该是 [B, 9, H, W])")
                    else:
                        # 9通道模型使用原来的方式
                        sampler_kwargs['test_model_kwargs'] = test_model_kwargs

                    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                    # breakpoint()
                    # ⭐ 修复2：使用处理后的ref_imgs和正确的数据类型（参考训练代码 ddpm_sft_13ch.py:3085-3099）
                    samples_ddim, intermediates = sampler.sample(S=opt.ddim_steps,
                                                        conditioning=c,
                                                        batch_size=test_batch.shape[0],
                                                        shape=shape,
                                                        verbose=False,
                                                        unconditional_guidance_scale=opt.scale,
                                                        unconditional_conditioning=uc,
                                                        eta=opt.ddim_eta,
                                                        x_T=start_code,
                                                        log_every_t=100,
                                                        src_im=ref_imgs_for_cond.to(torch.float32),
                                                        tar=test_batch.to(torch.float32),
                                                        **sampler_kwargs)
                    # breakpoint()
                    save_intermediates=False
                    
                    
                    # breakpoint()
                    if save_intermediates:
                        intermediate_pred_x0=intermediates['pred_x0']
                        intermediate_noised=intermediates['x_inter']
                        for i in range(len(intermediate_pred_x0)):
                            save_sample_by_decode(intermediate_pred_x0[i],model,Base_path="Save_intermediates/pred_x0",segment_id_batch=segment_id_batch,intermediate_num=i)
                            save_sample_by_decode(intermediate_noised[i],model,Base_path="Save_intermediates/intermediate",segment_id_batch=segment_id_batch,intermediate_num=i)
                    
                    
                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                    x_checked_image=x_samples_ddim
                    x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

                    def un_norm(x):
                        return (x+1.0)/2.0
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

                    if not opt.skip_save:
                        for i,x_sample in enumerate(x_checked_image_torch):
                            
                            # ✅ 检查结果文件是否已存在，如果存在则跳过
                            result_file = os.path.join(result_path, segment_id_batch[i]+".png")
                            if os.path.exists(result_file):
                                print(f"⏭️  跳过已存在的样本: {segment_id_batch[i]}")
                                continue

                            all_img=[]
                            all_img.append(un_norm(test_batch[i]).cpu())
                            all_img.append(un_norm(inpaint_image[i]).cpu())
                            ref_img=test_model_kwargs['ref_imgs'].squeeze(1)
                            ref_img=Resize([512,512])(ref_img)
                            all_img.append(un_norm_clip(ref_img[i]).cpu())
                            all_img.append(x_sample)
                            grid = torch.stack(all_img, 0)
                            grid = make_grid(grid)
                            grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                            img = Image.fromarray(grid.astype(np.uint8))
                            img.save(os.path.join(grid_path, 'grid-'+segment_id_batch[i]+'.png'))
                            



                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            img = Image.fromarray(x_sample.astype(np.uint8))
                            img.save(os.path.join(result_path, segment_id_batch[i]+".png"))
                            
                            mask_save=255.*rearrange(un_norm(inpaint_mask[i]).cpu(), 'c h w -> h w c').numpy()
                            mask_save= cv2.cvtColor(mask_save,cv2.COLOR_GRAY2RGB)
                            mask_save = Image.fromarray(mask_save.astype(np.uint8))
                            mask_save.save(os.path.join(sample_path, segment_id_batch[i]+"_mask.png"))
                            GT_img=255.*rearrange(all_img[0], 'c h w -> h w c').numpy()
                            GT_img = Image.fromarray(GT_img.astype(np.uint8))
                            GT_img.save(os.path.join(sample_path, segment_id_batch[i]+"_GT.png"))
                            inpaint_img=255.*rearrange(all_img[1], 'c h w -> h w c').numpy()
                            inpaint_img = Image.fromarray(inpaint_img.astype(np.uint8))
                            inpaint_img.save(os.path.join(sample_path, segment_id_batch[i]+"_inpaint.png"))
                            ref_img=255.*rearrange(all_img[2], 'c h w -> h w c').numpy()
                            ref_img = Image.fromarray(ref_img.astype(np.uint8))
                            ref_img.save(os.path.join(sample_path, segment_id_batch[i]+"_ref.png"))
                            base_count += 1



                    if not opt.skip_grid:
                        all_samples.append(x_checked_image_torch)

    print(f"\n{'='*80}")
    print(f"✅ Inference Complete!")
    print(f"📊 Processed: {processed_count} samples (indices {start_idx} to {end_idx-1})")
    print(f"📁 Results saved to: {outpath}")
    print(f"{'='*80}\n")
    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()

