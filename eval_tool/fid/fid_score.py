import os
import pathlib
import hashlib
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
import torch
import torchvision.transforms as TF
from PIL import Image
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

try:
    from .inception import InceptionV3
except ImportError:
    from inception import InceptionV3

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch-size', type=int, default=50,
                    help='Batch size to use')
parser.add_argument('--num-workers', type=int,
                    help=('Number of processes to use for data loading. '
                          'Defaults to `min(8, num_cpus)`'))
parser.add_argument('--device', type=str, default=None,
                    help='Device to use. Like cuda, cuda:0 or cpu')
parser.add_argument('--dims', type=int, default=2048,
                    choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                    help=('Dimensionality of Inception features to use. '
                          'By default, uses pool3 features'))
parser.add_argument('--max-samples', type=int, default=None,
                    help='Maximum number of generated samples to evaluate (does not apply to real images). If None, evaluate all generated samples.')
parser.add_argument('--stats-dir', type=str, default=None,
                    help=('Directory to save/load precomputed statistics. '
                          'If None, statistics will be saved in a .fid_stats subdirectory '
                          'next to the image directory'))
parser.add_argument('--save-stats', action='store_true',
                    help='Force save statistics even if cache exists')
parser.add_argument('--output', type=str, default=None,
                    help='Path to output text file to save FID score. If not specified, only print to console.')
parser.add_argument('path', type=str, nargs=2,
                    default=['dataset/FaceData/CelebAMask-HQ/CelebA-HQ-img', 'results/test_bench/results'],
                    help=('Paths to the generated images or '
                          'to .npz statistic files'))

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        # ⭐ 标准pytorch-fid做法：只ToTensor到[0,1]，让InceptionV3自己归一化
        # InceptionV3会在forward中执行: x = 2 * x - 1 (归一化到[-1,1])
        if transforms is None:
            self.transforms = TF.ToTensor()
        else:
            self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert('RGB')
        img = self.transforms(img)
        return img


def get_activations(files, model, batch_size=50, dims=2048, device='cpu',
                    num_workers=1):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    if batch_size > len(files):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(files)

    # 使用默认的InceptionV3预处理（在ImagePathDataset中定义）
    dataset = ImagePathDataset(files, transforms=None)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=num_workers)

    # ⭐ 修复：使用正确的维度（dims参数）
    pred_arr = np.empty((len(files), dims))

    start_idx = 0

    for batch in tqdm(dataloader):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        # 验证维度匹配
        if pred.shape[1] != dims:
            raise ValueError(f"Feature dimension mismatch: expected {dims}, got {pred.shape[1]}")

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(files, model, batch_size=50, dims=2048,
                                    device='cpu', num_workers=1):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(files, model, batch_size, dims, device, num_workers)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def get_stats_cache_path(image_path, max_samples, dims, stats_dir=None):
    """Generate cache file path for statistics.
    
    Args:
        image_path: Path to image directory
        max_samples: Maximum number of samples (None means all)
        dims: Feature dimensionality
        stats_dir: Directory to save stats (None means use default)
    
    Returns:
        Path to cache file
    """
    path_obj = pathlib.Path(image_path)
    
    # Generate a unique identifier based on path, max_samples, and dims
    try:
        path_str = str(path_obj.resolve())
    except (OSError, RuntimeError):
        # If resolve fails, use the absolute path as is
        path_str = str(path_obj.absolute())
    
    cache_key = f"{path_str}_{max_samples}_{dims}"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:12]
    
    # Determine stats directory
    if stats_dir is None:
        # Use .fid_stats subdirectory next to the image directory
        if path_obj.exists():
            stats_dir = path_obj.parent / '.fid_stats'
        else:
            # If path doesn't exist, use current directory
            stats_dir = pathlib.Path('.') / '.fid_stats'
    else:
        stats_dir = pathlib.Path(stats_dir)
    
    # Create stats directory if it doesn't exist
    stats_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate cache filename
    path_name = path_obj.name or 'root'
    cache_filename = f"{path_name}_{cache_hash}.npz"
    cache_path = stats_dir / cache_filename
    
    return cache_path


def compute_statistics_of_path(path, model, batch_size, dims, device,
                               num_workers=1, max_samples=None, 
                               stats_dir=None, save_stats=False):
    """Compute statistics for a path, with automatic caching.
    
    Args:
        path: Path to image directory or .npz file
        model: Model instance
        batch_size: Batch size
        dims: Feature dimensionality
        device: Device to use
        num_workers: Number of workers
        max_samples: Maximum number of samples
        stats_dir: Directory to save/load cached statistics 
                   (None means use default location, False means disable caching)
        save_stats: Force save statistics even if cache exists
    
    Returns:
        Tuple of (mu, sigma) statistics
    """
    if path.endswith('.npz'):
        # Directly load from .npz file
        with np.load(path) as f:
            m, s = f['mu'][:], f['sigma'][:]
        return m, s
    
    # Path to image directory
    image_path = pathlib.Path(path)
    
    # Generate cache file path only if caching is enabled
    cache_path = None
    if stats_dir is not False:  # False means disable caching
        cache_path = get_stats_cache_path(str(image_path), max_samples, dims, stats_dir)
        
        # Try to load from cache if it exists and we're not forcing save
        if not save_stats and cache_path.exists():
            print(f"Loading cached statistics from: {cache_path}")
            with np.load(cache_path) as f:
                m, s = f['mu'][:], f['sigma'][:]
            return m, s
    
    # Compute statistics
    files = sorted([file for ext in IMAGE_EXTENSIONS
                   for file in image_path.glob('*.{}'.format(ext))])
    
    # 限制评估样本数量
    total_files = len(files)
    if max_samples is not None and len(files) > max_samples:
        files = files[:max_samples]
        print(f"Limiting evaluation to first {max_samples} samples (out of {total_files})")
    
    print(f"Computing statistics for {len(files)} images...")
    m, s = calculate_activation_statistics(files, model, batch_size,
                                           dims, device, num_workers)
    
    # Save statistics to cache if caching is enabled
    if cache_path is not None:
        print(f"Saving statistics to cache: {cache_path}")
        np.savez(cache_path, mu=m, sigma=s)
    
    return m, s


def calculate_fid_given_paths(paths, batch_size, device, dims, num_workers=1, 
                              max_samples=None, stats_dir=None, save_stats=False,
                              cache_first_only=True):
    """Calculates the FID of two paths
    
    Args:
        paths: List of two paths (real images, generated images)
        max_samples: Maximum number of generated images to evaluate (does NOT apply to real images)
        cache_first_only: If True, only cache statistics for the first path (real images)
    """
    for p in paths:
        if not os.path.exists(p) and not p.endswith('.npz'):
            raise RuntimeError('Invalid path: %s' % p)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx]).to(device)

    # Compute statistics for first path (real images) - always use caching
    # max_samples does NOT apply to real images - always use all samples
    m1, s1 = compute_statistics_of_path(paths[0], model, batch_size,
                                        dims, device, num_workers, None,  # None = use all samples
                                        stats_dir, save_stats)
    
    # Compute statistics for second path (generated images)
    # max_samples only applies to generated images
    # By default, only cache the first path (real images) to save time
    # since generated images change frequently
    if cache_first_only:
        # Disable caching for generated images
        m2, s2 = compute_statistics_of_path(paths[1], model, batch_size,
                                            dims, device, num_workers, max_samples,  # Apply max_samples here
                                            False, False)  # stats_dir=False disables caching
    else:
        m2, s2 = compute_statistics_of_path(paths[1], model, batch_size,
                                            dims, device, num_workers, max_samples,  # Apply max_samples here
                                            stats_dir, save_stats)
    
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value


def main():
    args = parser.parse_args()

    if args.device is None:
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device(args.device)

    if args.num_workers is None:
        num_avail_cpus = len(os.sched_getaffinity(0))
        num_workers = min(num_avail_cpus, 8)
    else:
        num_workers = args.num_workers

    fid_value = calculate_fid_given_paths(args.path,
                                          args.batch_size,
                                          device,
                                          args.dims,
                                          num_workers,
                                          args.max_samples,
                                          args.stats_dir,
                                          args.save_stats)
    
    # 输出结果到控制台
    result_str = f'FID: {fid_value:.6f}'
    print(result_str)
    
    # 如果指定了输出文件，保存结果到文件
    if args.output:
        output_path = pathlib.Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(f"FID Score Evaluation Results\n")
            f.write(f"{'='*60}\n")
            f.write(f"Real images path: {args.path[0]}\n")
            f.write(f"Generated images path: {args.path[1]}\n")
            f.write(f"Batch size: {args.batch_size}\n")
            f.write(f"Feature dimensions: {args.dims}\n")
            f.write(f"Max samples (generated only): {args.max_samples if args.max_samples else 'All'}\n")
            f.write(f"Note: max_samples only applies to generated images, not real images.\n")
            f.write(f"{'='*60}\n")
            f.write(f"\nFID Score: {fid_value:.6f}\n")
        
        print(f"FID score saved to: {output_path}")


if __name__ == '__main__':
    main()


"""

CUDA_VISIBLE_DEVICES=3
python /data5/shuangjun.du/work/REFace/eval_tool/fid/fid_score.py \
    /data5/shuangjun.du/work/REFace/dataset/FaceData/CelebAMask-HQ/CelebA-HQ-img \
    /data5/shuangjun.du/work/REFace/results/CelebA/REFace/results \
    --max-samples 1000 \
    --save-stats \
    --output tmp/fid_score_celeba_1000.txt


CUDA_VISIBLE_DEVICES=3
python /data5/shuangjun.du/work/REFace/eval_tool/fid/fid_score.py \
    /data5/shuangjun.du/work/REFace/dataset/FaceData/FFHQ/images512 \
    /data5/shuangjun.du/work/REFace/results/FFHQ/REFace/results\
    --max-samples 1000 \
    --save-stats \
    --output tmp/fid_score_ffhq_1000.txt 
"""