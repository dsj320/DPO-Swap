import os
import pathlib
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
import torch
import torchvision.transforms as TF
from PIL import Image
import re
import torch.nn.functional as F
import eval_tool.face_vid2vid.modules.hopenet as hopenet1
from torchvision import models
import torchvision
try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x
from natsort import natsorted

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch-size', type=int, default=20,
                    help='Batch size to use')
parser.add_argument('--num-workers', type=int,
                    help=('Number of processes to use for data loading. '
                          'Defaults to `min(8, num_cpus)`'))
parser.add_argument('--device', type=str, default=None,
                    help='Device to use. Like cuda, cuda:0 or cpu')
parser.add_argument('--max-samples', type=int, default=None,
                    help='Maximum number of samples to evaluate. If None, evaluate all samples.')
parser.add_argument('--output', type=str, default=None,
                    help='Output file path to save results. If None, only print to console.')
parser.add_argument('path', type=str, nargs=2,
                    default=['dataset/FaceData/CelebAMask-HQ/CelebA-HQ-img', 'results/test_bench/results'],
                    help='Paths to the source and generated images')

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}

class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files):
        self.files = files
        self.transform_hopenet = torchvision.transforms.Compose([
            TF.ToTensor(),
            TF.Resize(size=(224, 224)),
            TF.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        image = self.transform_hopenet(Image.open(path).convert('RGB'))
        return image

def headpose_pred_to_degree(pred):
    device = pred.device
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).to(device)
    pred = F.softmax(pred)
    degree = torch.sum(pred*idx_tensor, axis=1) * 3 - 99

    return degree

def compute_features(files, model, batch_size=50, device='cpu', num_workers=1):
    """Calculates pose predictions (yaw, pitch, roll) for all images.
    Params:
    -- files       : List of image files paths
    -- model       : Hopenet model instance
    -- batch_size  : Batch size of images for the model to process at once
    -- device      : Device to run calculations
    -- num_workers  : Number of parallel dataloader workers
    Returns:
    -- A numpy array of dimension (num images, 3) containing (yaw, pitch, roll) angles
    """
    model.eval()

    if batch_size > len(files):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(files)

    dataset = ImagePathDataset(files)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=num_workers)

    pred_arr = np.empty((len(files),3))

    start_idx = 0
    
   
    for batch in tqdm(dataloader):
        batch = batch.to(device)
        with torch.no_grad():
            yaw_gt, pitch_gt, roll_gt = model(batch)
            yaw_gt = headpose_pred_to_degree(yaw_gt)
            pitch_gt = headpose_pred_to_degree(pitch_gt)
            roll_gt = headpose_pred_to_degree(roll_gt)
        
        yaw_gt = yaw_gt.cpu().numpy().reshape(-1, 1)
        pitch_gt = pitch_gt.cpu().numpy().reshape(-1, 1)
        roll_gt = roll_gt.cpu().numpy().reshape(-1, 1)
        pred_arr[start_idx:start_idx + yaw_gt.shape[0]] = np.concatenate((yaw_gt, pitch_gt, roll_gt), axis=1)

        start_idx = start_idx + yaw_gt.shape[0]

    return pred_arr


def compute_features_wrapp(path, model, batch_size, device,
                               num_workers=1, max_samples=None):
    path = pathlib.Path(path)
    files = natsorted([file for ext in IMAGE_EXTENSIONS
                   for file in path.glob('*.{}'.format(ext))])
    
    # 限制评估样本数量
    if max_samples is not None and len(files) > max_samples:
        print(f"Limiting Pose evaluation to first {max_samples} samples (out of {len(files)})")
        files = files[:max_samples]
    
    # Extract numbers from filenames for indexing
    pattern = r'[_\/.-]'
    parts = [re.split(pattern, str(file.name)) for file in files]
    numbers = [[int(par) for par in part if par.isdigit()] for part in parts]
    numbers = [num[-1] for num in numbers if len(num) > 0]
    mi_num = min(numbers)
    numbers = [(num - mi_num) for num in numbers]
    
    pred_arr = compute_features(files, model, batch_size, device, num_workers)

    return pred_arr, numbers


def calculate_id_given_paths(paths, batch_size, device, num_workers=1, max_samples=None):
    """Calculates the pose distance between two image paths"""
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)
    
    hopenet = hopenet1.Hopenet(models.resnet.Bottleneck, [3, 4, 6, 3], 66)
    print('Loading hopenet')
    hopenet_state_dict = torch.load('Other_dependencies/Hopenet_pose/hopenet_robust_alpha1.pkl')
    hopenet.load_state_dict(hopenet_state_dict)
    if torch.cuda.is_available():
        hopenet = hopenet.cuda()
    hopenet.eval()

    feat1, ori_lab = compute_features_wrapp(paths[0], hopenet, batch_size,
                                        device, num_workers, max_samples)
    feat2, swap_lab = compute_features_wrapp(paths[1], hopenet, batch_size,
                                        device, num_workers, max_samples)
    
    feat1 = feat1[swap_lab]
    # Calculate L2 distance between corresponding poses
    dist = np.linalg.norm(feat1 - feat2, axis=1)
    Value = np.mean(dist)
    
    return Value


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

    Pose_value = calculate_id_given_paths(args.path,
                                          args.batch_size,
                                          device,
                                          num_workers,
                                          args.max_samples)
    
    # Prepare output
    output_lines = []
    output_lines.append('Pose_value: {:.6f}'.format(Pose_value))
    
    # Print to console
    for line in output_lines:
        print(line)
    
    # Write to file if output path is specified
    if args.output:
        output_path = pathlib.Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for line in output_lines:
                f.write(line + '\n')
        print(f'\nResults saved to: {output_path}')


if __name__ == '__main__':
    main()


"""
CUDA_VISIBLE_DEVICES=3 \
PYTHONPATH="/data5/shuangjun.du/work/REFace:$PYTHONPATH" \
    python /data5/shuangjun.du/work/REFace/eval_tool/Pose/pose_compare.py \
    /data5/shuangjun.du/work/REFace/dataset/FaceData/FFHQ/Val_target \
    /data5/shuangjun.du/work/REFace/results/FFHQ/REFace/results \
    --max-samples 200 \
    --output tmp/pose_compare_ffhq_200.txt


CUDA_VISIBLE_DEVICES=3 \
PYTHONPATH="/data5/shuangjun.du/work/REFace:$PYTHONPATH" \
    python /data5/shuangjun.du/work/REFace/eval_tool/Pose/pose_compare.py \
    /data5/shuangjun.du/work/REFace/dataset/FaceData/CelebAMask-HQ/Val_512 \
    /data5/shuangjun.du/work/REFace/results/CelebA/REFace/results \
    --max-samples 1000 \
    --output tmp/pose_compare_celeba_1000.txt
"""