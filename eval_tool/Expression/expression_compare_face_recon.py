import os
import pathlib
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
import torch
import torchvision.transforms as TF
from PIL import Image
import re
import torchvision
try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x
from natsort import natsorted

from eval_tool.Deep3DFaceRecon_pytorch_edit.options.test_options import TestOptions

# give empty string to use the default options
test_opt = TestOptions('')
test_opt = test_opt.parse()

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
parser.add_argument('--print_sim', type=bool, default=False,
                    help='Whether to print individual similarity values.')
parser.add_argument('path', type=str, nargs=2,
                    default=['dataset/FaceData/CelebAMask-HQ/Val_target', 'results_grad/v4_reconstruct_img_train_2_step_multi_false_with_LPIPS_ep16/results'],
                    help='Paths to the source and generated images')



IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}

from eval_tool.Deep3DFaceRecon_pytorch_edit.models import create_model


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files):
        self.files = files
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        im = Image.open(path).convert('RGB')
        # Resize to 512
        im = im.resize((512, 512), Image.BICUBIC)
        im = torch.tensor(np.array(im)/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        return im

def compute_features(files, model, batch_size=50, device='cpu', num_workers=1):
    """Calculates expression coefficients for all images.
    Params:
    -- files       : List of image files paths
    -- model       : Deep3DFaceRecon model instance
    -- batch_size  : Batch size of images for the model to process at once
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers
    Returns:
    -- A numpy array of dimension (num images, 64) containing expression coefficients
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

    pred_arr = np.empty((len(files), 64))
    start_idx = 0
    
    for batch in tqdm(dataloader):
        batch = batch.to(device).squeeze(1)

        with torch.no_grad():
            coeff = model.forward(batch)
            pred = coeff['exp']
        
        pred = pred.cpu().numpy()
        pred_arr[start_idx:start_idx + pred.shape[0]] = pred
        start_idx = start_idx + pred.shape[0]

    return pred_arr


def compute_features_wrapp(path, model, batch_size, device,
                               num_workers=1, max_samples=None):
    path = pathlib.Path(path)
    files = natsorted([file for ext in IMAGE_EXTENSIONS
                   for file in path.glob('*.{}'.format(ext))])
    
    # 限制评估样本数量
    if max_samples is not None and len(files) > max_samples:
        print(f"Limiting Expression evaluation to first {max_samples} samples (out of {len(files)})")
        files = files[:max_samples]
    
    # Use position indices instead of file name numbers to ensure consistency
    # Files are already sorted, so position index [0, 1, 2, ..., N-1] is correct
    numbers = list(range(len(files)))
    
    pred_arr = compute_features(files, model, batch_size, device, num_workers)

    return pred_arr, numbers


def calculate_id_given_paths(paths, batch_size, device, num_workers=1, max_samples=None):
    """Calculates the expression distance between two image paths"""
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)
    
    models_expression = create_model(test_opt)
    models_expression.setup(test_opt)
    
    if torch.cuda.is_available():
        models_expression.net_recon.cuda()
        models_expression.facemodel.to("cuda")
    models_expression.eval()

    feat1, ori_lab = compute_features_wrapp(paths[0], models_expression, batch_size,
                                        device, num_workers, max_samples)
    feat2, swap_lab = compute_features_wrapp(paths[1], models_expression, batch_size,
                                        device, num_workers, max_samples)
    
    feat1 = feat1[swap_lab]
    
    # Calculate L2 distance between corresponding expressions
    diff_feat = np.power(feat1 - feat2, 2)
    diff_feat = np.sum(diff_feat, axis=-1)
    Value = np.sqrt(diff_feat)
    similarities = Value
    Value = np.mean(Value)
    
    return Value, similarities


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

    Expression_value, similarities = calculate_id_given_paths(args.path,
                                          args.batch_size,
                                          device,
                                          num_workers,
                                          args.max_samples)
    
    # Prepare output
    output_lines = []
    output_lines.append('Expression_value: {:.6f}'.format(Expression_value))
    
    if args.print_sim:
        output_lines.append('Similarities: \n ')
        for i in range(len(similarities)):
            output_lines.append('{}: {:.6f}'.format(i, similarities[i]))
    
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
    python /data5/shuangjun.du/work/REFace/eval_tool/Expression/expression_compare_face_recon.py \
    /data5/shuangjun.du/work/REFace/dataset/FaceData/FFHQ/Val_target \
    /data5/shuangjun.du/work/REFace/results/FFHQ/REFace/results \
    --max-samples 200 \
    --output tmp/expression_compare_ffhq_200.txt
"""