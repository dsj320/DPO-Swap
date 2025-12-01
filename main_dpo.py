import argparse, os, sys, datetime, glob, importlib, csv
import numpy as np
import time
import torch
import torchvision
import pytorch_lightning as pl

from packaging import version
from omegaconf import OmegaConf
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from functools import partial
from PIL import Image

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info

from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.util import instantiate_from_config
import socket
from pytorch_lightning.plugins.environments import ClusterEnvironment,SLURMEnvironment
import wandb
wandb.login(key="f0a412d675fd5439a95ac8369fe5fe7b6acf6fc7")

def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        # default="models/REFace/checkpoints/last.ckpt",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=["configs/debug.yaml"],
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=True,
        nargs="?",
        help="Is train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "--train_cross_attn_only",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train cross attn only",
    )
    parser.add_argument(
        "-p",
        "--project",
        help="name of new or path to existing project"
    )
    
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="models/REFace/Debug",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="checkpoints/model.ckpt",
        help="path to pretrained model",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )
    parser.add_argument(
        "--train_from_scratch",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Train from scratch",
    )
    return parser


def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id

    if isinstance(dataset, Txt2ImgIterableBaseDataset):
        split_size = dataset.num_records // worker_info.num_workers
        # reset num_records to the true number to retain reliable length information
        dataset.sample_ids = dataset.valid_ids[worker_id * split_size:(worker_id + 1) * split_size]
        current_id = np.random.choice(len(np.random.get_state()[1]), 1)
        return np.random.seed(np.random.get_state()[1][current_id] + worker_id)
    else:
        return np.random.seed(np.random.get_state()[1][0] + worker_id)



class DataModuleFromConfig(pl.LightningDataModule):    
    def __init__(self, batch_size, train=None, validation=None, test=None, predict=None,
                 wrap=False, num_workers=None, shuffle_test_loader=False, use_worker_init_fn=False,
                 shuffle_val_dataloader=False):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        self.wrap = wrap

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False if is_iterable_dataset else True,
                          worker_init_fn=init_fn)

    def _val_dataloader(self, shuffle=False):
        if isinstance(self.datasets['validation'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          worker_init_fn=init_fn,
                          shuffle=shuffle)

    def _test_dataloader(self, shuffle=False):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        # do not shuffle dataloader for iterable dataset
        shuffle = shuffle and (not is_iterable_dataset)

        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn, shuffle=shuffle)

    def _predict_dataloader(self, shuffle=False):
        if isinstance(self.datasets['predict'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["predict"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn)


class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_keyboard_interrupt(self, trainer, pl_module):
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    def on_pretrain_routine_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            if "callbacks" in self.lightning_config:
                if 'metrics_over_trainsteps_checkpoint' in self.lightning_config['callbacks']:
                    os.makedirs(os.path.join(self.ckptdir, 'trainstep_checkpoints'), exist_ok=True)
            print("Project config")
            print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            print("Lightning config")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass


class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = {
            pl.loggers.TestTubeLogger: self._testtube,
            pl.loggers.WandbLogger: self._wandb,
        }
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w

            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(
                tag, grid,
                global_step=pl_module.global_step)

    @rank_zero_only
    def _wandb(self, pl_module, images, batch_idx, split):
        """记录图像到 wandb - 拼接成 2 行布局（处理不同尺寸）"""
        try:
            print(f"[_wandb] Called with {len(images)} images: {list(images.keys())}")
            
            if wandb.run is None:
                print("WARNING: wandb.run is None, skipping image logging")
                return
            
            from PIL import Image as PILImage
            import torch.nn.functional as F
            
            # 定义布局：
            # 第 1 行：src, tgt, winner
            # 第 2 行：output_reference, output_current, loser
            row1_keys = ['src', 'tgt', 'winner']
            row2_keys = ['output_reference', 'output_current', 'loser']
            
            def resize_tensor_to_512(tensor):
                """将 tensor resize 到 512x512"""
                # tensor: [B, C, H, W]
                if tensor.shape[2] == 512 and tensor.shape[3] == 512:
                    return tensor
                print(f"    Resizing from {tensor.shape[2]}x{tensor.shape[3]} to 512x512")
                return F.interpolate(tensor, size=(512, 512), mode='bilinear', align_corners=False)
            
            def create_row_grid(keys_list):
                """为一行创建拼接的 grid"""
                row_tensors = []
                for k in keys_list:
                    if k in images:
                        img_tensor = images[k].detach().cpu()
                        # 统一 resize 到 512
                        img_tensor = resize_tensor_to_512(img_tensor)
                        row_tensors.append(img_tensor)
                        print(f"  Added {k}: shape={img_tensor.shape}")
                
                if not row_tensors:
                    return None
                
                # 拼接所有类别的图片
                all_imgs = torch.cat(row_tensors, dim=0)
                
                # 每个类别有 N 个样本，横向排列
                num_samples_per_category = images[keys_list[0]].shape[0] if keys_list[0] in images else 4
                
                # 创建 grid: 每类一行，横向显示该类的所有样本
                grid = torchvision.utils.make_grid(
                    all_imgs,
                    nrow=num_samples_per_category,
                    normalize=True,
                    value_range=(-1, 1),
                    padding=2
                )
                
                # 转换为 PIL
                grid_np = grid.permute(1, 2, 0).numpy()
                grid_np = np.clip(grid_np, 0, 1)
                grid_np = (grid_np * 255).astype(np.uint8)
                return PILImage.fromarray(grid_np)
            
            # 生成两行
            print(f"[_wandb] Creating row 1: {row1_keys}")
            pil_row1 = create_row_grid(row1_keys)
            
            print(f"[_wandb] Creating row 2: {row2_keys}")
            pil_row2 = create_row_grid(row2_keys)
            
            if pil_row1 is None and pil_row2 is None:
                print("[_wandb] No valid images to create grid")
                return
            
            # 垂直拼接两行
            if pil_row1 is not None and pil_row2 is not None:
                max_width = max(pil_row1.width, pil_row2.width)
                total_height = pil_row1.height + pil_row2.height
                
                final_img = PILImage.new('RGB', (max_width, total_height), (255, 255, 255))
                final_img.paste(pil_row1, (0, 0))
                final_img.paste(pil_row2, (0, pil_row1.height))
                
                print(f"[_wandb] Final grid: {final_img.size} (row1: {pil_row1.size}, row2: {pil_row2.size})")
                
            elif pil_row1 is not None:
                final_img = pil_row1
                print(f"[_wandb] Only row 1: {final_img.size}")
            else:
                final_img = pil_row2
                print(f"[_wandb] Only row 2: {final_img.size}")
            
            # 记录到 wandb - 只记录一个拼接后的大图
            wandb_log = {
                f"{split}/all_samples": wandb.Image(
                    final_img,
                    caption=f"Step {pl_module.global_step} | Row1: {', '.join(row1_keys)} | Row2: {', '.join(row2_keys)}"
                )
            }
            
            wandb.log(wandb_log, commit=True)
            print(f"✓ Successfully logged grid to wandb")
            
            # 强制同步
            try:
                if hasattr(wandb.run, '_file_pusher') and wandb.run._file_pusher:
                    wandb.run._file_pusher.push()
                print(f"✓ Forced sync to wandb server")
            except Exception as sync_err:
                print(f"⚠ Sync warning: {sync_err}")
                
        except Exception as e:
            print(f"✗ ERROR logging images to wandb: {e}")
            import traceback
            traceback.print_exc()

    @rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k,
                global_step,
                current_epoch,
                batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)
            
            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)
            
            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            # 记录到 PyTorch Lightning logger（testtube）
            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, pl_module.global_step, split)
            
            # 同时记录到 wandb（无论用什么 logger）
            self._wandb(pl_module, images, batch_idx, split)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and (
                check_idx > 0 or self.log_first_step):
            try:
                self.log_steps.pop(0)
            except IndexError as e:
                print(e)
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        print(f"[ImageLogger.on_train_batch_end] Called at global_step={pl_module.global_step}, batch_idx={batch_idx}, disabled={self.disabled}")
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
            print(f"[ImageLogger.on_train_batch_end] Calling log_img...")
            self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        print(f"[ImageLogger.on_validation_batch_end] Called at global_step={pl_module.global_step}, batch_idx={batch_idx}, disabled={self.disabled}")
        if not self.disabled and pl_module.global_step > 0:
            print(f"[ImageLogger.on_validation_batch_end] Calling log_img...")
            self.log_img(pl_module, batch, batch_idx, split="val")
        if hasattr(pl_module, 'calibrate_grad_norm'):
            if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)
    
    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module, unused=None):
        """每个 epoch 结束时强制同步 wandb"""
        try:
            import wandb
            if wandb.run is not None:
                print(f"[ImageLogger] Syncing wandb at epoch end...")
                # 强制刷新所有待上传的数据
                if hasattr(wandb.run, '_file_pusher') and wandb.run._file_pusher:
                    wandb.run._file_pusher.push()
                if hasattr(wandb.run, '_backend') and wandb.run._backend:
                    wandb.run._backend.interface.publish_files()
                print(f"[ImageLogger] Wandb sync complete")
        except Exception as e:
            print(f"[ImageLogger] Wandb sync failed: {e}")


class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.root_gpu)
        torch.cuda.synchronize(trainer.root_gpu)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        torch.cuda.synchronize(trainer.root_gpu)
        max_memory = torch.cuda.max_memory_allocated(trainer.root_gpu) / 2 ** 20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.training_type_plugin.reduce(max_memory)
            epoch_time = trainer.training_type_plugin.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass


if __name__ == "__main__":

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    
    sys.path.append(os.getcwd())

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)

    opt, unknown = parser.parse_known_args()
    
    if opt.debug:
        # set cuda visible devices =3
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        
    
    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            # idx = len(paths)-paths[::-1].index("logs")+1
            # logdir = "/".join(paths[:idx])
            logdir = "/".join(paths[:-2])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
        else:
            name = ""
        nowname = now + name + opt.postfix
        logdir = os.path.join(opt.logdir, nowname)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed)

    # try:
        # init and save configs
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    lightning_config = config.pop("lightning", OmegaConf.create())
    # merge trainer cli with config
    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    # default to ddp
    trainer_config["accelerator"] = "ddp"
    for k in nondefault_trainer_args(opt):
        trainer_config[k] = getattr(opt, k)
    if not "gpus" in trainer_config and not "devices" in trainer_config:
        del trainer_config["accelerator"]
        cpu = True
    elif "devices" in trainer_config:
        print("Using devices:", trainer_config["devices"])
        print("using nodes:", trainer_config["num_nodes"])
        trainer_config["accelerator"] = "gpu"
        trainer_config["strategy"] = "ddp"
        trainer_config["gpus"] = "0, 1, 2, 3"
        cpu = False
    else:
        gpuinfo = trainer_config["gpus"]
        print(f"Running on GPUs {gpuinfo}")
        cpu = False
    trainer_opt = argparse.Namespace(**trainer_config)
    lightning_config.trainer = trainer_config
    
    # 处理wandb resume逻辑
    if opt.resume:
        # 如果是resume训练，使用"allow"模式，wandb会尝试恢复或创建新run
        wandb_resume = "allow"
        wandb_id = nowname  # 使用相同的ID来恢复run
    else:
        # 新训练，不resume
        wandb_resume = None
        wandb_id = nowname
    
    # 从配置文件读取 wandb 配置
    wandb_config = lightning_config.get("wandb", OmegaConf.create())
    wandb_project = wandb_config.get("project", "Face_Swapping_Debug" if opt.debug else "Face_Swapping")
    wandb_run_name = wandb_config.get("run_name", nowname) or nowname  # 如果为None则使用nowname
    wandb_tags = wandb_config.get("tags", [])
    wandb_notes = wandb_config.get("notes", "")
    
    print(f"[WANDB Config] Project: {wandb_project}, Run Name: {wandb_run_name}")
    if wandb_tags:
        print(f"[WANDB Config] Tags: {wandb_tags}")
    
    # 手动初始化 wandb（原始方式）- 只在主进程
    from pytorch_lightning.utilities import rank_zero_only
    import torch.distributed as dist
    
    # 检查当前进程的 rank
    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0
    
    if rank == 0:
        # 配置 wandb settings - 强制在线模式并立即同步
        wandb_settings = wandb.Settings(
            mode="online",         # ✅ 强制在线模式
            start_method="fork",   # 多进程兼容
            _disable_stats=False,  # 启用系统统计
            _disable_meta=False,   # 启用元数据
            _save_requirements=False,  # 不保存 requirements
            _file_stream_timeout_seconds=30,  # 文件流超时
            # _file_pusher_timeout_seconds=30,  # 已移除：旧版本参数，不再支持
        )
        
        # 确保 wandb 目录有写权限
        wandb_dir = os.path.join(logdir, "wandb")
        os.makedirs(wandb_dir, exist_ok=True)
        
        # 测试写权限
        try:
            test_file = os.path.join(wandb_dir, ".test_write")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            print(f"[WANDB] Using wandb dir: {wandb_dir}")
        except Exception as e:
            print(f"[WANDB] WARNING: {wandb_dir} not writable ({e}), using /tmp")
            wandb_dir = "/tmp/wandb_logs"
            os.makedirs(wandb_dir, exist_ok=True)
        
        # 统一初始化 wandb，使用配置文件中的参数
        wandb.init(
            project=wandb_project, 
            name=wandb_run_name, 
            tags=list(wandb_tags) if wandb_tags else None,
            notes=wandb_notes if wandb_notes else None,
            config=vars(opt), 
            dir=wandb_dir, 
            resume=wandb_resume, 
            id=wandb_id,
            settings=wandb_settings,
            reinit=False
        )
        
        # 验证 wandb 初始化
        print(f"[WANDB Rank {rank}] wandb.run is {'INITIALIZED' if wandb.run is not None else 'None (ERROR!)'}")
        if wandb.run is not None:
            print(f"[WANDB] Run name: {wandb.run.name}, ID: {wandb.run.id}")
            print(f"[WANDB] URL: {wandb.run.url}")
            print(f"[WANDB] Mode: {wandb.run.mode}")
            print(f"[WANDB] Current step: {wandb.run.step}")
    else:
        print(f"[WANDB Rank {rank}] Skipping wandb.init() - not main process")
    
    print(config)
    

    # model
    model = instantiate_from_config(config.model)
    
    # ------------------- 修改后的加载逻辑开始 -------------------
    if not opt.resume:
        # 这是“开始新DPO训练”的逻辑
        print(f"Loading base model for NEW DPO training from: {opt.pretrained_model}")
        if not os.path.exists(opt.pretrained_model):
            raise FileNotFoundError(f"Cannot find pretrained model at {opt.pretrained_model}")

        # 1. 加载基础模型 (e.g., sd-v1-4.ckpt) 的 state dict
        base_sd = torch.load(opt.pretrained_model, map_location='cpu')['state_dict']
        
        # 2. 创建一个新的 state_dict，用于同时填充 policy_model (model) 和 ref_model (model_ref)
        dpo_state_dict = {}
        
        print("Copying base weights to policy (model.*) and reference (model_ref.*)...")
        
        total_copied_to_ref = 0
        
        # 3. 遍历基础模型的所有权重
        for key, value in base_sd.items():
            
            # 3a. 将权重按原样复制. 这会填充:
            # - self.first_stage_model.*
            # - self.cond_stage_model.*
            # - self.model.* (即 策略模型/Policy Model)
            dpo_state_dict[key] = value

            # 3b. 检查这个键是否属于 UNet (根据你的错误日志, UNet 键以 "model.diffusion_model" 开头)
            unet_prefix = "model.diffusion_model"
            if key.startswith(unet_prefix):
                
                # 3c. 为参考模型(Reference Model)创建对应的键
                # 例如: "model.diffusion_model.X" -> "model_ref.diffusion_model.X"
                # (注意: "model." 被替换为 "model_ref.")
                ref_key = "model_ref." + key[len("model."):] 
                
                # 3d. 为参考模型添加权重的 *副本*
                dpo_state_dict[ref_key] = value.clone()
                total_copied_to_ref += 1

        print(f"Total keys in base model: {len(base_sd)}")
        print(f"Total keys copied to ref_model (UNet only): {total_copied_to_ref}")

        # 4. 将这个合并后的 state_dict 加载到你的 LatentDiffusion 模型中
        #    使用 strict=False 是常规操作，因为基础模型 state_dict 包含 VAE 和 CLIP，
        #    而你的 LatentDiffusion 类本身没有直接定义它们 (它们在 first_stage_model 等子模块中)
        missing_keys, unexpected_keys = model.load_state_dict(dpo_state_dict, strict=False)
        
        print(f"Successfully loaded weights for DPO.")
        print(f"  Missing Keys: {len(missing_keys)}")
        print(f"  Unexpected Keys: {len(unexpected_keys)}")
        
        if total_copied_to_ref == 0:
            print("\n\n*** 严重警告 ***")
            print("在您的预训练模型中没有找到任何 'model.diffusion_model' 开头的键。")
            print("这意味着您的 'model_ref' (参考模型) 没有被正确初始化，DPO 训练将失败。")
            print("请检查您的 --pretrained_model 文件是否正确。")
            print("************************\n\n")

    else:
        # 这是“恢复DPO训练”的逻辑
        # (当你使用 -r 时，trainer.fit() 会自动加载检查点, 无需额外代码)
        print(f"Resuming existing DPO training from DPO checkpoint: {opt.resume_from_checkpoint}")
    
    # ------------------- 修改后的加载逻辑结束 -------------------

    trainer_kwargs = dict()

    # default logger configs
    # 使用 testtube 作为 PyTorch Lightning logger，wandb 通过手动 init 管理
    default_logger_cfgs = {
        "wandb": {
            "target": "pytorch_lightning.loggers.WandbLogger",
            "params": {
                "project": "Face_Swapping_Debug" if opt.debug else "Face_Swapping",
                "name": nowname,
                "save_dir": logdir,
                "offline": opt.debug,
                "id": wandb_id,
                "resume": wandb_resume,
            }
        },
        "testtube": {
            "target": "pytorch_lightning.loggers.TestTubeLogger",
            "params": {
                "name": "testtube",
                "save_dir": logdir,
            }
        },
    }
    # 使用 testtube logger，wandb 单独管理
    default_logger_cfg = default_logger_cfgs["testtube"]
    if "logger" in lightning_config:
        logger_cfg = lightning_config.logger
    else:
        logger_cfg = OmegaConf.create()
    logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
    trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

    # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
    # specify which metric is used to determine best models
    default_modelckpt_cfg = {
        "target": "pytorch_lightning.callbacks.ModelCheckpoint",
        "params": {
            "dirpath": ckptdir,
            "filename": "{epoch:06}",
            "verbose": True,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
            "save_last": True,
        }
    }
    if hasattr(model, "monitor"):
        print(f"Monitoring {model.monitor} as checkpoint metric.")
        default_modelckpt_cfg["params"]["monitor"] = model.monitor
        default_modelckpt_cfg["params"]["save_top_k"] = 30

    if "modelcheckpoint" in lightning_config:
        modelckpt_cfg = lightning_config.modelcheckpoint
    else:
        modelckpt_cfg =  OmegaConf.create()
    modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
    print(f"Merged modelckpt-cfg: \n{modelckpt_cfg}")
    if version.parse(pl.__version__) < version.parse('1.4.0'):
        trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)

    # add callback which sets up log directory
    default_callbacks_cfg = {
        "setup_callback": {
            "target": "main_dpo.SetupCallback",
            "params": {
                "resume": opt.resume,
                "now": now,
                "logdir": logdir,
                "ckptdir": ckptdir,
                "cfgdir": cfgdir,
                "config": config,
                "lightning_config": lightning_config,
            }
        },
        "image_logger": {
            "target": "main_dpo.ImageLogger",
            "params": {
                "batch_frequency":50,
                "max_images": 3,
                "clamp": True
            }
        },
        "learning_rate_logger": {
            "target": "pytorch_lightning.callbacks.LearningRateMonitor",
            "params": {
                "logging_interval": "step",
                # "log_momentum": True
            }
        },
        "cuda_callback": {
            "target": "main_dpo.CUDACallback"
        },
    }
    if version.parse(pl.__version__) >= version.parse('1.4.0'):
        default_callbacks_cfg.update({'checkpoint_callback': modelckpt_cfg})

    if "callbacks" in lightning_config:
        callbacks_cfg = lightning_config.callbacks
    else:
        callbacks_cfg = OmegaConf.create()

    if 'metrics_over_trainsteps_checkpoint' in callbacks_cfg:
        print(
            'Caution: Saving checkpoints every n train steps without deleting. This might require some free space.')
        default_metrics_over_trainsteps_ckpt_dict = {
            'metrics_over_trainsteps_checkpoint':
                {"target": 'pytorch_lightning.callbacks.ModelCheckpoint',
                    'params': {
                        "dirpath": os.path.join(ckptdir, 'trainstep_checkpoints'),
                        "filename": "{epoch:06}-{step:09}",
                        "verbose": True,
                        'save_top_k': -1,
                        'every_n_train_steps': 10000,
                        'save_weights_only': True
                    }
                    }
        }
        default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)

    callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
    if 'ignore_keys_callback' in callbacks_cfg and hasattr(trainer_opt, 'resume_from_checkpoint'):
        callbacks_cfg.ignore_keys_callback.params['ckpt_path'] = trainer_opt.resume_from_checkpoint
    elif 'ignore_keys_callback' in callbacks_cfg:
        del callbacks_cfg['ignore_keys_callback']

    trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]

    trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
    # trainer.plugins = [MyCluster()]
    trainer.logdir = logdir  ###

    # data
    data = instantiate_from_config(config.data)
    # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
    # calling these ourselves should not be necessary but it is.
    # lightning still takes care of proper multiprocessing though
    data.prepare_data()
    data.setup()
    print("#### Data #####")
    for k in data.datasets:
        print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

    # configure learning rate
    bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
    if not cpu:
        ngpu = len(lightning_config.trainer.gpus.strip(",").split(','))
    else:
        ngpu = 1
    if 'accumulate_grad_batches' in lightning_config.trainer:
        accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
    else:
        accumulate_grad_batches = 1
    # if 'num_nodes' in lightning_config.trainer:
    #     num_nodes = lightning_config.trainer.num_nodes
    # else:
    num_nodes = 1
    print(f"accumulate_grad_batches = {accumulate_grad_batches}")
    lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
    if opt.scale_lr:
        model.learning_rate = accumulate_grad_batches * num_nodes * ngpu * bs * base_lr
        print(
            "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_nodes) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                model.learning_rate, accumulate_grad_batches, num_nodes, ngpu, bs, base_lr))
    else:
        model.learning_rate = base_lr
        print("++++ NOT USING LR SCALING ++++")
        print(f"Setting learning rate to {model.learning_rate:.2e}")


    # allow checkpointing via USR1
    def melk(*args, **kwargs):
        # run all checkpoint hooks
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)


    def divein(*args, **kwargs):
        if trainer.global_rank == 0:
            import pudb;
            pudb.set_trace()


    import signal

    signal.signal(signal.SIGUSR1, melk)
    signal.signal(signal.SIGUSR2, divein)

    # run
    if opt.train:
        try:
            trainer.fit(model, data)
        except Exception:
            melk()
            raise
    if not opt.no_test and not trainer.interrupted:
        trainer.test(model, data)
