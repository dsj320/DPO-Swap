import argparse, os, sys, datetime, glob
import numpy as np
import time
import torch
import torchvision
import pytorch_lightning as pl
import shutil

from packaging import version
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from functools import partial
from PIL import Image

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info

from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.util import instantiate_from_config
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
        default=["configs/train_dpo.yaml"],  # 修改默认值为train_dpo.yaml
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
                          worker_init_fn=init_fn,
                          persistent_workers=True if self.num_workers > 0 else False)

    def _val_dataloader(self, shuffle=False):
        if isinstance(self.datasets['validation'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          worker_init_fn=init_fn,
                          shuffle=shuffle,
                          persistent_workers=True if self.num_workers > 0 else False)

    def _test_dataloader(self, shuffle=False):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        # do not shuffle dataloader for iterable dataset
        shuffle = shuffle and (not is_iterable_dataset)

        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn, shuffle=shuffle,
                          persistent_workers=True if self.num_workers > 0 else False)

    def _predict_dataloader(self, shuffle=False):
        if isinstance(self.datasets['predict'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["predict"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn,
                          persistent_workers=True if self.num_workers > 0 else False)


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
    
    def save_code_snapshot(self):
        """保存代码快照到日志目录"""
        # 创建代码快照目录
        snapshot_dir = os.path.join(self.logdir, "code_snapshot")
        os.makedirs(snapshot_dir, exist_ok=True)
        
        # 项目根目录（相对于当前工作目录）
        project_root = "/data5/shuangjun.du/work/REFace"
        
        # 需要保存的文件列表（相对路径）
        files_to_snapshot = [
            "train_sft.sh",
            "main_dpo.py",
            "ldm/data/dpo_dataset.py",
            "ldm/models/diffusion/ddpm_dpo.py",
        ]
        
        print("\n" + "="*80)
        print("📸 Creating code snapshot...")
        print("="*80)
        
        for rel_path in files_to_snapshot:
            src_file = os.path.join(project_root, rel_path)
            
            # 检查源文件是否存在
            if not os.path.exists(src_file):
                print(f"⚠️  Warning: {rel_path} not found, skipping...")
                continue
            
            # 在快照目录中创建相同的子目录结构
            dst_file = os.path.join(snapshot_dir, rel_path)
            dst_dir = os.path.dirname(dst_file)
            os.makedirs(dst_dir, exist_ok=True)
            
            # 复制文件
            try:
                shutil.copy2(src_file, dst_file)
                print(f"✓ Saved: {rel_path}")
            except Exception as e:
                print(f"✗ Failed to save {rel_path}: {e}")
        
        # 保存当前时间戳
        timestamp_file = os.path.join(snapshot_dir, "snapshot_info.txt")
        with open(timestamp_file, 'w') as f:
            f.write(f"Snapshot created at: {self.now}\n")
            f.write(f"Training started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Working directory: {os.getcwd()}\n")
            f.write(f"\nSnapshot files:\n")
            for rel_path in files_to_snapshot:
                f.write(f"  - {rel_path}\n")
        
        print(f"✓ Snapshot info saved to: {os.path.relpath(timestamp_file, os.getcwd())}")
        print("="*80 + "\n")

    def on_pretrain_routine_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            if "callbacks" in self.lightning_config:
                if 'metrics_over_trainsteps_checkpoint' in self.lightning_config['callbacks']:
                    os.makedirs(os.path.join(self.ckptdir, 'trainstep_checkpoints'), exist_ok=True)
            
            # 📸 保存代码快照
            self.save_code_snapshot()
            
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
        """记录图像到 wandb - 拼接成 2 行布局（处理不同尺寸）- 非阻塞版本"""
        try:
            print(f"[_wandb] Called with {len(images)} images: {list(images.keys())}")
            
            if wandb.run is None:
                print("WARNING: wandb.run is None, skipping image logging")
                return
            
            from PIL import Image as PILImage
            import torch.nn.functional as F
            
            # 根据是否有参考模型输出，定义要显示的图像顺序（每行一种类型）
            has_reference = 'output_reference' in images
            
            if has_reference:
                # DPO 模式：有参考模型（5行）
                # 第 1 行：src
                # 第 2 行：tgt
                # 第 3 行：winner
                # 第 4 行：loser
                # 第 5 行：output_reference
                # 第 6 行：output_current
                row_keys_list = ['src', 'tgt', 'winner', 'loser', 'output_reference', 'output_current']
            else:
                # SFT 模式：无参考模型（4行）
                # 第 1 行：src
                # 第 2 行：tgt
                # 第 3 行：winner
                # 第 4 行：output_current
                row_keys_list = ['src', 'tgt', 'winner', 'output_current']
            
            def resize_tensor_to_512(tensor):
                """将 tensor resize 到 512x512"""
                # tensor: [B, C, H, W]
                if tensor.shape[2] == 512 and tensor.shape[3] == 512:
                    return tensor
                print(f"    Resizing from {tensor.shape[2]}x{tensor.shape[3]} to 512x512")
                return F.interpolate(tensor, size=(512, 512), mode='bilinear', align_corners=False)
            
            def create_single_row_grid(key):
                """为单个类型创建一行 grid（横向显示所有样本）"""
                if key not in images:
                    return None
                
                img_tensor = images[key].detach().cpu()
                # 统一 resize 到 512
                img_tensor = resize_tensor_to_512(img_tensor)
                
                print(f"  Creating row for {key}: shape={img_tensor.shape}")
                
                # 创建 grid: 所有样本横向排列
                grid = torchvision.utils.make_grid(
                    img_tensor,
                    nrow=img_tensor.shape[0],  # 所有样本放在一行
                    normalize=True,
                    value_range=(-1, 1),
                    padding=2
                )
                
                # 转换为 PIL
                grid_np = grid.permute(1, 2, 0).numpy()
                grid_np = np.clip(grid_np, 0, 1)
                grid_np = (grid_np * 255).astype(np.uint8)
                return PILImage.fromarray(grid_np)
            
            # 生成每一行
            pil_rows = []
            for key in row_keys_list:
                print(f"[_wandb] Creating row for: {key}")
                pil_row = create_single_row_grid(key)
                if pil_row is not None:
                    pil_rows.append(pil_row)
            
            if not pil_rows:
                print("[_wandb] No valid images to create grid")
                return
            
            # 垂直拼接所有行
            max_width = max(row.width for row in pil_rows)
            total_height = sum(row.height for row in pil_rows)
            
            final_img = PILImage.new('RGB', (max_width, total_height), (255, 255, 255))
            
            current_y = 0
            for i, pil_row in enumerate(pil_rows):
                final_img.paste(pil_row, (0, current_y))
                current_y += pil_row.height
                print(f"[_wandb] Pasted row {i+1} at y={current_y - pil_row.height}")
            
            print(f"[_wandb] Final grid: {final_img.size} with {len(pil_rows)} rows")
            
            # ⭐ 关键修改：使用 commit=False 避免阻塞
            # wandb会在后台异步上传，不会阻塞训练
            # 生成描述
            mode = "DPO" if has_reference else "SFT"
            rows_desc = " | ".join([f"Row{i+1}: {key}" for i, key in enumerate(row_keys_list) if key in images])
            caption = f"Step {pl_module.global_step} | {mode} Mode | {rows_desc}"
            
            # 在键名中包含步数，使文件名更清晰
            wandb_log = {
                f"{split}/all_samples_step_{pl_module.global_step:06d}": wandb.Image(
                    final_img,
                    caption=caption
                )
            }
            
            # ⭐ commit=False: 不立即同步，由wandb后台处理
            wandb.log(wandb_log, step=pl_module.global_step, commit=False)
            print(f"✓ Successfully queued image to wandb at step {pl_module.global_step} (non-blocking)")
                
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

    @rank_zero_only  # ← 修复：只在主进程执行，避免多GPU重复调用导致死锁
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        print(f"[ImageLogger.on_train_batch_end] Called at global_step={pl_module.global_step}, batch_idx={batch_idx}, disabled={self.disabled}")
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
            print(f"[ImageLogger.on_train_batch_end] Calling log_img...")
            self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        # ⭐ 禁用验证阶段的图片记录，避免 DDIM 采样导致验证过程过慢
        # 验证阶段的 log_images 会触发 50+ 步 DDIM 推理，非常耗时
        pass
        if hasattr(pl_module, 'calibrate_grad_norm'):
            if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)


class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.root_gpu)
        torch.cuda.synchronize(trainer.root_gpu)
        self.start_time = time.time()
        print(f"\n[CUDACallback] 🚀 Starting Epoch {pl_module.current_epoch}...")

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

    # init and save configs
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    lightning_config = config.pop("lightning", OmegaConf.create())
    # merge trainer cli with config
    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    for k in nondefault_trainer_args(opt):
        trainer_config[k] = getattr(opt, k)

    # 与 main.py 保持一致：只要提供了 gpus/devices 就启用 GPU；否则退回 CPU
    if "gpus" in trainer_config or "devices" in trainer_config:
        # PyTorch Lightning 1.4.2 使用 gpus 而不是 devices
        # 确保使用正确的参数名称
        num_gpus = None
        if "gpus" in trainer_config:
            g = trainer_config["gpus"]
            if isinstance(g, str):
                gpu_ids = [x for x in g.replace(" ", "").split(",") if x != ""]
                num_gpus = len(gpu_ids) if gpu_ids else 1
            elif isinstance(g, int):
                num_gpus = g if g > 0 else 1
            else:
                num_gpus = 1
        elif "devices" in trainer_config:
            # 如果配置中使用了 devices，将其转换为 gpus（PL 1.4.2兼容）
            num_gpus = trainer_config["devices"]
            trainer_config["gpus"] = num_gpus
            # 删除 devices 参数，因为 PL 1.4.2 不支持
            del trainer_config["devices"]
        
        # PL 1.4.2 不需要显式设置 accelerator='gpu'，有 gpus 参数就够了
        # 如果显式设置 accelerator，可能会导致冲突
        if "accelerator" in trainer_config:
            del trainer_config["accelerator"]
        
        # 多卡时默认 ddp；单卡则不强制 distributed_backend
        # pytorch-lightning 1.4.2 使用 distributed_backend 而不是 strategy
        if num_gpus and num_gpus > 1:
            trainer_config.setdefault("distributed_backend", "ddp")
        cpu = False
        print(f"Using GPU training with gpus={trainer_config.get('gpus', num_gpus)}")
    else:
        trainer_config.pop("accelerator", None)
        trainer_config.pop("devices", None)
        cpu = True
        print("Running on CPU")

    # 从 trainer_config 中提取 distributed_backend，避免传递给 Trainer.from_argparse_args
    # pytorch-lightning 1.4.2 中 distributed_backend 应该通过 kwargs 传递，而不是 argparse
    distributed_backend_value = trainer_config.pop("distributed_backend", None)
    
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
    import torch.distributed as dist
    
    # 检查当前进程的 rank
    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0
    
    if rank == 0:
        # ⭐ 方案选择说明：
        # - WANDB_MODE="disabled": 完全禁用wandb（推荐，避免阻塞）
        # - WANDB_MODE="offline": 离线模式，数据保存本地（可后续手动同步）
        # - WANDB_MODE="online": 在线模式（已优化非阻塞，但仍可能有网络延迟）
        
        wandb_mode = os.environ.get("WANDB_MODE", "online")
        print(f"[WANDB Rank {rank}] Mode: {wandb_mode}")
        
        # ⭐ 配置 wandb settings - 非阻塞模式
        wandb_settings = wandb.Settings(
            mode=wandb_mode,       # 使用环境变量控制模式
            start_method="fork",   # 多进程兼容
            _disable_stats=False,  # 启用系统统计
            _disable_meta=False,   # 启用元数据
            _save_requirements=False,  # 不保存 requirements
            _file_stream_timeout_seconds=30,  # 文件流超时
            _stats_sample_rate_seconds=30,  # 降低统计采样频率
            _stats_samples_to_average=10,  # 统计样本平均数
        )
        
        print(f"[WANDB Rank {rank}] Initializing with mode={wandb_mode}")
    else:
        print(f"[WANDB Rank {rank}] Skipping wandb init (not main process)")
        # ⭐ 重要：非主进程完全不使用wandb，避免DDP死锁
        os.environ["WANDB_MODE"] = "disabled"
        wandb_mode = "disabled"
    
    # ⭐ 只有rank 0才初始化wandb
    if rank == 0:
        # 确保 wandb 目录有写权限
        wandb_dir = os.path.join(logdir, "wandb")
        os.makedirs(wandb_dir, exist_ok=True)
        
        # 测试写权限
        try:
            test_file = os.path.join(wandb_dir, ".test_write")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            print(f"[WANDB Rank {rank}] Using wandb dir: {wandb_dir}")
        except Exception as e:
            print(f"[WANDB Rank {rank}] WARNING: {wandb_dir} not writable ({e}), using /tmp")
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
        # 这是"开始新DPO训练"的逻辑
        print(f"Loading base model for NEW training from: {opt.pretrained_model}")
        if not os.path.exists(opt.pretrained_model):
            raise FileNotFoundError(f"Cannot find pretrained model at {opt.pretrained_model}")

        # 1. 加载基础模型 (e.g., sd-v1-4.ckpt) 的 state dict
        base_sd = torch.load(opt.pretrained_model, map_location='cpu')['state_dict']
        
        # 2. 创建一个新的 state_dict
        dpo_state_dict = {}
        
        # 检查是否使用 SFT 模式
        use_sft_mode = config.model.params.get('use_sft_loss', False)
        
        if use_sft_mode:
            print("✓ SFT 模式：只加载策略模型权重（跳过参考模型，节省显存）")
        else:
            print("DPO 模式：加载策略模型和参考模型权重...")
        
        total_copied_to_ref = 0
        
        # 3. 遍历基础模型的所有权重
        for key, value in base_sd.items():
            
            # 3a. 将权重按原样复制. 这会填充:
            # - self.first_stage_model.*
            # - self.cond_stage_model.*
            # - self.model.* (即 策略模型/Policy Model)
            dpo_state_dict[key] = value

            # 3b. ⭐ 只在 DPO 模式下复制参考模型权重
            if not use_sft_mode:
                # 检查这个键是否属于 UNet (根据你的错误日志, UNet 键以 "model.diffusion_model" 开头)
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
        if not use_sft_mode:
            print(f"Total keys copied to ref_model (UNet only): {total_copied_to_ref}")
        else:
            print(f"SFT 模式：跳过参考模型权重复制")

        # 4. 将这个合并后的 state_dict 加载到你的 LatentDiffusion 模型中
        #    使用 strict=False 是常规操作，因为基础模型 state_dict 包含 VAE 和 CLIP，
        #    而你的 LatentDiffusion 类本身没有直接定义它们 (它们在 first_stage_model 等子模块中)
        missing_keys, unexpected_keys = model.load_state_dict(dpo_state_dict, strict=False)
        
        print(f"Successfully loaded weights.")
        print(f"  Missing Keys: {len(missing_keys)}")
        print(f"  Unexpected Keys: {len(unexpected_keys)}")
        
        # ⭐ 只在 DPO 模式下检查参考模型
        if not use_sft_mode and total_copied_to_ref == 0:
            print("\n\n*** 严重警告 ***")
            print("在您的预训练模型中没有找到任何 'model.diffusion_model' 开头的键。")
            print("这意味着您的 'model_ref' (参考模型) 没有被正确初始化，DPO 训练将失败。")
            print("请检查您的 --pretrained_model 文件是否正确。")
            print("************************\n\n")

    else:
        # 这是"恢复训练"的逻辑
        # (当你使用 -r 时，trainer.fit() 会自动加载检查点, 无需额外代码)
        print(f"Resuming training from checkpoint: {opt.resume_from_checkpoint}")
    
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
                "clamp": True,
                "log_first_step": False  # 避免第一步就记录图像，提升启动速度
            }
        },
        "learning_rate_logger": {
            "target": "pytorch_lightning.callbacks.LearningRateMonitor",
            "params": {
                "logging_interval": "step",
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

    # PyTorch Lightning 1.4.2 使用 gpus 而不是 devices，不需要显式传递 accelerator
    # gpus 参数已经在 trainer_config 中，会通过 trainer_opt 传递
    # 确保不传递 PL 1.4.2 不支持的参数
    if "devices" in trainer_kwargs:
        del trainer_kwargs["devices"]
    if "accelerator" in trainer_kwargs:
        del trainer_kwargs["accelerator"]
    
    # pytorch-lightning 1.4.2 使用 distributed_backend 而不是 strategy
    # 通过 kwargs 传递 distributed_backend，而不是通过 argparse
    if distributed_backend_value is not None:
        trainer_kwargs["distributed_backend"] = distributed_backend_value

    trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
    trainer.logdir = logdir

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
        # 处理 gpus 参数可能是字符串或整数的情况
        gpus_param = lightning_config.trainer.gpus
        if isinstance(gpus_param, str):
            ngpu = len(gpus_param.strip(",").split(','))
        elif isinstance(gpus_param, int):
            ngpu = gpus_param if gpus_param > 0 else 1
        else:
            ngpu = 1
    else:
        ngpu = 1
    if 'accumulate_grad_batches' in lightning_config.trainer:
        accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
    else:
        accumulate_grad_batches = 1
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
    import atexit
    import threading

    # 创建一个标志来跟踪是否已经保存过checkpoint（避免重复保存）
    _checkpoint_saved = threading.Lock()
    _saving_checkpoint = False

    def safe_save_checkpoint():
        """安全地保存checkpoint，避免重复保存"""
        global _saving_checkpoint
        with _checkpoint_saved:
            if _saving_checkpoint:
                return  # 已经在保存中，跳过
            _saving_checkpoint = True
        
        try:
            # 检查trainer是否已初始化且是主进程
            if trainer is not None:
                # 在多进程环境下，只有rank 0保存
                if hasattr(trainer, 'global_rank'):
                    if trainer.global_rank == 0:
                        print("\n[Checkpoint] Saving checkpoint...")
                        melk()
                        print("[Checkpoint] Checkpoint saved successfully.")
                else:
                    # 单进程环境
                    print("\n[Checkpoint] Saving checkpoint...")
                    melk()
                    print("[Checkpoint] Checkpoint saved successfully.")
        except Exception as e:
            print(f"[Checkpoint] Error saving checkpoint: {e}")
            import traceback
            traceback.print_exc()
        finally:
            with _checkpoint_saved:
                _saving_checkpoint = False

    # 注册退出时的清理函数，确保进程退出时保存checkpoint
    def save_checkpoint_on_exit():
        """进程退出时保存checkpoint"""
        safe_save_checkpoint()
    
    # 注册atexit处理函数（注意：SIGKILL无法被捕获，但SIGTERM可以）
    atexit.register(save_checkpoint_on_exit)

    # 处理SIGTERM信号（kill命令默认发送的信号）
    def sigterm_handler(signum, frame):
        """处理SIGTERM信号（kill命令）"""
        print(f"\n[SIGTERM] Received termination signal (kill)")
        safe_save_checkpoint()
        print("[SIGTERM] Exiting...")
        # 使用os._exit强制退出，避免被其他信号处理干扰
        os._exit(0)
    
    # 处理SIGINT信号（Ctrl+C），确保能够中断
    def sigint_handler(signum, frame):
        """处理SIGINT信号（Ctrl+C）"""
        print(f"\n[SIGINT] Received interrupt signal (Ctrl+C)")
        safe_save_checkpoint()
        print("[SIGINT] Exiting...")
        # 使用os._exit强制退出
        os._exit(0)

    signal.signal(signal.SIGUSR1, melk)
    signal.signal(signal.SIGUSR2, divein)
    signal.signal(signal.SIGTERM, sigterm_handler)  # 处理kill命令
    signal.signal(signal.SIGINT, sigint_handler)    # 处理Ctrl+C

    # run
    if opt.train:
        try:
            print(f"[TRAINING] Starting training with {trainer.max_epochs} epochs...")
            print(f"[TRAINING] Current callbacks: {[type(cb).__name__ for cb in trainer.callbacks]}")
            # pytorch-lightning 1.4.2 使用 distributed_backend 而不是 strategy
            ddp_info = getattr(trainer, 'distributed_backend', None) or getattr(trainer, 'strategy', 'N/A')
            print(f"[TRAINING] DDP enabled: {ddp_info}")
            
            trainer.fit(model, data)
            
            print(f"[TRAINING] Training completed successfully!")
        except KeyboardInterrupt:
            # ⭐ 专门处理 Ctrl+C（备用，如果信号处理失败）
            print("\n[KeyboardInterrupt] Training interrupted by user (Ctrl+C)")
            try:
                if trainer.global_rank == 0:
                    melk()  # 保存checkpoint
                print("Checkpoint saved. Exiting...")
            except Exception as e:
                print(f"Error saving checkpoint: {e}")
            sys.exit(0)
        except Exception:
            # 其他异常也保存
            try:
                if trainer.global_rank == 0:
                    melk()
            except:
                pass
            raise
    if not opt.no_test and not trainer.interrupted:
        trainer.test(model, data)
