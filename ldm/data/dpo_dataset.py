from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import imp

import os
from io import BytesIO
import json
import logging
import base64
from sys import prefix
import threading
import random
from turtle import left, right
import numpy as np
from typing import Any, Callable, List, Tuple, Union
from PIL import Image,ImageDraw
import torch.utils.data as data
import json
import time
import cv2
import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as T
import copy
import math
from functools import partial
import albumentations as A
import bezier

import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from einops import rearrange
from torchvision.utils import save_image



import warnings
warnings.filterwarnings("ignore")


from torchvision.transforms import ToTensor, ToPILImage
# sys.path.append('/data/pzb/EAT/attack')
from thinplatespline.batch import TPS
from thinplatespline.tps import tps_warp
TOTEN = ToTensor()
TOPIL = ToPILImage()
DEVICE = torch.device("cpu")



def grid_points_2d(width, height, device=DEVICE):
    """
    Create 2d grid points. Distribute points across a width and height,
    with the resulting coordinates constrained to -1, 1
    returns tensor shape (width * height, 2)
    """
    xx, yy = torch.meshgrid(
        [torch.linspace(-1.0, 1.0, height, device=device),
         torch.linspace(-1.0, 1.0, width, device=device)])
    return torch.stack([yy, xx], dim=-1).contiguous().view(-1, 2)
def noisy_grid(width, height, noise_matrix, device=DEVICE):
    """
    Make uniform grid points, and add noise except for edge points.
    """
    grid = grid_points_2d(width, height, device)
    mod = torch.zeros([height, width, 2], device=device)
    mod[1:height - 1, 1:width - 1, :] = noise_matrix
    return grid + mod.reshape(-1, 2)
def grid_to_img(grid_points, width, height):
    """
    convert (N * 2) tensor of grid points in -1, 1 to tuple of (x, y)
    scaled to width, height.
    return (x, y) to plot"""
    grid_clone = grid_points.clone().detach().cpu().numpy()
    x = (1 + grid_clone[..., 0]) * (width - 1) / 2
    y = (1 + grid_clone[..., 1]) * (height - 1) / 2
    return x.flatten(), y.flatten()
def decow(img,scale=0.8):
    n, c, w, h = img.size()
    device = torch.device('cpu')
    a = 3
    X = grid_points_2d(a, a, device)
    noise = (torch.rand([a-2, a-2, 2]) - 0.5) * scale
    # noise = (torch.rand([1, 1, 2]) - 0.5)
    Y = noisy_grid(a, a, noise, device)
    tpsb = TPS(size=(h, w), device=device)
    warped_grid_b = tpsb(X[None, ...], Y[None, ...])
    warped_grid_b = warped_grid_b.repeat(img.shape[0], 1, 1, 1)
    awt_img = torch.grid_sampler_2d(img, warped_grid_b, 0, 0, False)
    return awt_img


def bbox_process(bbox):
    x_min = int(bbox[0])
    y_min = int(bbox[1])
    x_max = x_min + int(bbox[2])
    y_max = y_min + int(bbox[3])
    return list(map(int, [x_min, y_min, x_max, y_max]))


def get_tensor(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return torchvision.transforms.Compose(transform_list)

def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)


#####

# 1:skin, 2:nose, 3:eye_g, 4:l_eye, 5:r_eye, 6:l_brow, 7:r_brow, 8:l_ear, 9:r_ear, 
# 10:mouth, 11:u_lip, 12:l_lip, 13:hair, 14:hat, 15:ear_r, 16:neck_l, 17:neck, 18:cloth

# 19 attributes in total, skin-1,nose-2,...cloth-18, background-0
celelbAHQ_label_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye',
                        'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth',
                        'u_lip', 'l_lip', 'hair', 'hat', 'ear_r',
                        'neck_l', 'neck', 'cloth']

# face-parsing.PyTorch also includes 19 attributes，but with different permutation
face_parsing_PyTorch_label_list = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye',
                                    'eye_g', 'l_ear', 'r_ear', 'ear_r', 'nose', 
                                    'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 
                                    'cloth', 'hair', 'hat']  # skin-1 l_brow-2 ...
 
# 9 attributes with left-right aggrigation
faceParser_label_list = ['background', 'mouth', 'eyebrows', 'eyes', 'hair', 
                         'nose', 'skin', 'ears', 'belowface']

# 12 attributes with left-right aggrigation
faceParser_label_list_detailed = ['background', 'lip', 'eyebrows', 'eyes', 'hair', 
                                  'nose', 'skin', 'ears', 'belowface', 'mouth', 
                                  'eye_glass', 'ear_rings']

TO_TENSOR = transforms.ToTensor()
MASK_CONVERT_TF = transforms.Lambda(
    lambda celebAHQ_mask: __celebAHQ_masks_to_faceParser_mask(celebAHQ_mask))

MASK_CONVERT_TF_DETAILED = transforms.Lambda(
    lambda celebAHQ_mask: __celebAHQ_masks_to_faceParser_mask_detailed(celebAHQ_mask))


NORMALIZE = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

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

def get_transforms(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)



def __celebAHQ_masks_to_faceParser_mask_detailed(celebA_mask):
    """Convert the semantic image of CelebAMaskHQ to reduced categories (12-class). 

    Args:
        mask (PIL image): with shape [H,W]
    Return:
        aggrigated mask, with same shape [H,W] but the number of segmentation classes is less
    """
    # 19 attributes in total, skin-1,nose-2,...cloth-18, background-0
    celelbAHQ_label_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye',
                            'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth',
                            'u_lip', 'l_lip', 'hair', 'hat', 'ear_r',
                            'neck_l', 'neck', 'cloth']# 12 attributes with left-right aggrigation
    faceParser_label_list_detailed = ['background', 'lip', 'eyebrows', 'eyes', 'hair', 
                                    'nose', 'skin', 'ears', 'belowface', 'mouth', 
                                  'eye_glass', 'ear_rings']

    converted_mask = np.zeros_like(celebA_mask)

    backgorund = np.equal(celebA_mask, 0)
    converted_mask[backgorund] = 0

    lip = np.logical_or(np.equal(celebA_mask, 11), np.equal(celebA_mask, 12))
    converted_mask[lip] = 1

    eyebrows = np.logical_or(np.equal(celebA_mask, 6),
                             np.equal(celebA_mask, 7))
    converted_mask[eyebrows] = 2

    eyes = np.logical_or(np.equal(celebA_mask, 4), np.equal(celebA_mask, 5))
    converted_mask[eyes] = 3

    hair = np.equal(celebA_mask, 13)
    converted_mask[hair] = 4

    nose = np.equal(celebA_mask, 2)
    converted_mask[nose] = 5

    skin = np.equal(celebA_mask, 1)
    # print('skin', np.sum(skin))
    converted_mask[skin] = 6

    ears = np.logical_or(np.equal(celebA_mask, 8), np.equal(celebA_mask, 9))
    converted_mask[ears] = 7

    belowface = np.equal(celebA_mask, 17)
    converted_mask[belowface] = 8
    
    mouth = np.equal(celebA_mask, 10)   
    converted_mask[mouth] = 9

    eye_glass = np.equal(celebA_mask, 3)
    converted_mask[eye_glass] = 10
    
    ear_rings = np.equal(celebA_mask, 15)
    converted_mask[ear_rings] = 11
    
    return converted_mask

def __celebAHQ_masks_to_faceParser_mask(celebA_mask):
    """Convert the semantic image of CelebAMaskHQ to reduced categories (9-class). 

    Args:
        mask (PIL image): with shape [H,W]
    Return:
        aggrigated mask, with same shape [H,W] but the number of segmentation classes is less
    """

    assert len(celebA_mask.size) == 2, "The provided mask should be with [H,W] format"

    converted_mask = np.zeros_like(celebA_mask)

    backgorund = np.equal(celebA_mask, 0)
    converted_mask[backgorund] = 0

    mouth = np.logical_or(
        np.logical_or(np.equal(celebA_mask, 10), np.equal(celebA_mask, 11)),
        np.equal(celebA_mask, 12)
    )
    converted_mask[mouth] = 1

    eyebrows = np.logical_or(np.equal(celebA_mask, 6),
                             np.equal(celebA_mask, 7))
    converted_mask[eyebrows] = 2

    eyes = np.logical_or(np.equal(celebA_mask, 4), np.equal(celebA_mask, 5))
    converted_mask[eyes] = 3

    hair = np.equal(celebA_mask, 13)
    converted_mask[hair] = 4

    nose = np.equal(celebA_mask, 2)
    converted_mask[nose] = 5

    skin = np.equal(celebA_mask, 1)
    converted_mask[skin] = 6

    ears = np.logical_or(np.equal(celebA_mask, 8), np.equal(celebA_mask, 9))
    converted_mask[ears] = 7

    belowface = np.equal(celebA_mask, 17)
    converted_mask[belowface] = 8

    return converted_mask




class CelebAdataset(data.Dataset):
    def __init__(self,state,arbitrary_mask_percent=0,load_vis_img=False,label_transform=None,fraction=1.0,**args
        ):
        self.label_transform=label_transform
        self.fraction=fraction
        self.load_vis_img=load_vis_img
        self.state=state
        self.args=args
        self.arbitrary_mask_percent=arbitrary_mask_percent
        self.kernel = np.ones((1, 1), np.uint8)
        self.random_trans=A.Compose([
            A.Resize(height=224,width=224),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=20),
            A.Blur(p=0.3),
            A.ElasticTransform(p=0.3), 
            # A.GaussNoise(p=0.3),# newly added from this line
            # A.HueSaturationValue(p=0.3),
            # A.ISONoise(p=0.3),
            # A.Solarize(p=0.3),
            ])
        
        self.gray_outer_mask=args['gray_outer_mask']
        # self.preserve=args['preserve_mask']
        if hasattr(args, 'preserve_mask'):
            self.preserve=args['preserve_mask']
            self.remove_tar=args['preserve_mask']
            self.preserve_src=args['preserve_mask']
        else:
            self.preserve=args['preserve_mask_src']
            self.remove_tar=args['remove_mask_tar']
            self.preserve_src=args['preserve_mask_src']
        
        
        self.Fullmask=False
        
        self.bbox_path_list=[]
        if state == "train":
            self.imgs = sorted([osp.join(args['dataset_dir'], "CelebA-HQ-img", "%d.jpg"%idx) for idx in range(28000)])
            # self.labels = ([osp.join(self.root, "CelebA-HQ-mask", "%d"%int(idx/2000) ,'{0:0=5d}'.format(idx)+'_skin.png') for idx in range(28000)])
            self.labels =  sorted([osp.join(args['dataset_dir'], "CelebA-HQ-mask/Overall_mask", "%d.png"%idx) for idx in range(28000)]) 
            self.labels_vis =  sorted([osp.join(args['dataset_dir'], "vis", "%d.png"%idx) for idx in range(28000)]) if self.load_vis_img else None
        elif state == "validation":
            self.imgs = sorted([osp.join(args['dataset_dir'], "CelebA-HQ-img", "%d.jpg"%idx) for idx in range(28000, 30000)])
            # self.labels = ([osp.join(self.root, "CelebA-HQ-mask", "%d"%int(idx/2000) ,'{0:0=5d}'.format(idx)+'_skin.png') for idx in range(28000, 30000)])
            self.labels =  sorted([osp.join(args['dataset_dir'], "CelebA-HQ-mask/Overall_mask", "%d.png"%idx) for idx in range(28000, 30000)]) 
            self.labels_vis =  sorted([osp.join(args['dataset_dir'], "vis", "%d.png"%idx) for idx in range(28000, 30000)]) if self.load_vis_img else None
        else:
            self.imgs = sorted([osp.join(args['dataset_dir'], "CelebA-HQ-img", "%d.jpg"%idx) for idx in range(28000, 30000)])
            # self.labels = ([osp.join(self.root, "CelebA-HQ-mask", "%d"%int(idx/2000) ,'{0:0=5d}'.format(idx)+'_skin.png') for idx in range(28000, 30000)])
            self.labels =  sorted([osp.join(args['dataset_dir'], "CelebA-HQ-mask/Overall_mask", "%d.png"%idx) for idx in range(28000, 30000)]) 
            self.labels_vis =  sorted([osp.join(args['dataset_dir'], "vis", "%d.png"%idx) for idx in range(28000, 30000)]) if self.load_vis_img else None
        
        self.imgs= self.imgs[:int(len(self.imgs)*self.fraction)]
        self.labels= self.labels[:int(len(self.labels)*self.fraction)]
        self.labels_vis= self.labels_vis[:int(len(self.labels_vis)*self.fraction)]  if self.load_vis_img else None

        if self.load_vis_img:
            assert len(self.imgs) == len(self.labels) == len(self.labels_vis)
        else:
            assert len(self.imgs) == len(self.labels)

        # image pairs indices
        self.indices = np.arange(len(self.imgs))
        self.length=len(self.indices)

    def __getitem__(self, index):
        if self.gray_outer_mask:
            return self.__getitem_gray__(index)
        else:
            return self.__getitem_black__(index)


    def __getitem_gray__(self, index):

        img_path = self.imgs[index]
        img_p = Image.open(img_path).convert('RGB')
     

        ############
        mask_path = self.labels[index]
        mask_img = Image.open(mask_path).convert('L')
        
        if self.Fullmask:
            mask_img_full=mask_img
            mask_img_full=get_tensor(normalize=False, toTensor=True)(mask_img_full)
        
        mask_img = np.array(mask_img)  # Convert the label to a NumPy array if it's not already
        
        
            
        
        # Create a mask to preserve values in the 'preserve' list
        # preserve = [1,2,4,5,8,9,17 ]
        # preserve = [1,2,4,5,8,9 ]
        preserve = self.preserve # full mask to be changed
        mask = np.isin(mask_img, preserve)

        # Create a converted_mask where preserved values are set to 255
        converted_mask = np.zeros_like(mask_img)
        converted_mask[mask] = 255
        # convert to PIL image
        mask_img=Image.fromarray(converted_mask).convert('L')
        mask_tensor=1-get_tensor(normalize=False, toTensor=True)(mask_img)
 
 

        if self.load_vis_img:
            label_vis = self.labels_vis[index]
            label_vis = Image.open(label_vis).convert('RGB')
            label_vis = TO_TENSOR(label_vis)
        else:
            label_vis = -1  # unified interface
        
    
        img_p_np=cv2.imread(img_path)
        img_p_np = cv2.cvtColor(img_p_np, cv2.COLOR_BGR2RGB)
        ref_image_tensor=img_p_np
        # resize mask_img
       
    
        
        # ref_image_tensor=self.random_trans(image=ref_image_tensor)
        ref_image_tensor=Image.fromarray(ref_image_tensor)
        ref_image_tensor=get_tensor_clip()(ref_image_tensor)
       

        ### Generate mask
        image_tensor = get_tensor()(img_p)
        W,H = img_p.size

        image_tensor_cropped=image_tensor
        mask_tensor_cropped=mask_tensor
        image_tensor_resize=T.Resize([self.args['image_size'],self.args['image_size']])(image_tensor_cropped)
        mask_tensor_resize=T.Resize([self.args['image_size'],self.args['image_size']])(mask_tensor_cropped)
        
        # a=random.randint(1,4)
        scale=random.uniform(0.5, 1.0)
        mask_tensor_resize=decow(mask_tensor_resize.unsqueeze(0) ,scale=scale).squeeze(0)
        inpaint_tensor_resize=image_tensor_resize*mask_tensor_resize
        
        mask_ref=1-T.Resize([1024,1024])(mask_tensor)
        ref_image_tensor=ref_image_tensor*mask_ref
        
        # ref_image_tensor=Image.fromarray(ref_image_tensor)
        ref_image_tensor=255.* rearrange(un_norm_clip(ref_image_tensor), 'c h w -> h w c').cpu().numpy()
        
        ref_image_tensor=self.random_trans(image=ref_image_tensor)
        ref_image_tensor=Image.fromarray(ref_image_tensor['image'].astype(np.uint8)) 
        ref_image_tensor=get_tensor_clip()(ref_image_tensor)
   
        if self.Fullmask:
            return {"GT":image_tensor_resize,"inpaint_image":inpaint_tensor_resize,"inpaint_mask":mask_img_full,"ref_imgs":ref_image_tensor}
   
        return {"GT":image_tensor_resize,"inpaint_image":inpaint_tensor_resize,"inpaint_mask":mask_tensor_resize,"ref_imgs":ref_image_tensor}

    def __getitem_black__(self, index):
        # black mask
        img_path = self.imgs[index]
        img_p = Image.open(img_path).convert('RGB')
     

        ############
        mask_path = self.labels[index]
        mask_img = Image.open(mask_path).convert('L')
        mask_img = np.array(mask_img)  # Convert the label to a NumPy array if it's not already

        # Create a mask to preserve values in the 'preserve' list
        # preserve = [1,2,4,5,8,9,17 ]
        # preserve = [1,2,4,5,8,9 ]
        preserve = self.preserve # full mask to be changed
        mask = np.isin(mask_img, preserve)

        # Create a converted_mask where preserved values are set to 255
        converted_mask = np.zeros_like(mask_img)
        converted_mask[mask] = 255
        # convert to PIL image
        mask_img=Image.fromarray(converted_mask).convert('L')
        mask_tensor=1-get_tensor(normalize=False, toTensor=True)(mask_img)
 
 

        if self.load_vis_img:
            label_vis = self.labels_vis[index]
            label_vis = Image.open(label_vis).convert('RGB')
            label_vis = TO_TENSOR(label_vis)
        else:
            label_vis = -1  # unified interface
        
    
        img_p_np=cv2.imread(img_path)
        img_p_np = cv2.cvtColor(img_p_np, cv2.COLOR_BGR2RGB)
        ref_image_tensor=img_p_np
        # resize mask_img
        mask_img_r = mask_img.resize(img_p_np.shape[1::-1], Image.NEAREST)
        mask_img_r = np.array(mask_img_r)
        
        # select only mask_img region from reference image
        ref_image_tensor[mask_img_r==0]=0   # comment this if full img should be used
    
        
        ref_image_tensor=self.random_trans(image=ref_image_tensor)
        ref_image_tensor=Image.fromarray(ref_image_tensor["image"])
        ref_image_tensor=get_tensor_clip()(ref_image_tensor)



        ### Generate mask
        image_tensor = get_tensor()(img_p)
        W,H = img_p.size

        image_tensor_cropped=image_tensor
        mask_tensor_cropped=mask_tensor
        image_tensor_resize=T.Resize([self.args['image_size'],self.args['image_size']])(image_tensor_cropped)
        mask_tensor_resize=T.Resize([self.args['image_size'],self.args['image_size']])(mask_tensor_cropped)
        inpaint_tensor_resize=image_tensor_resize*mask_tensor_resize
   
        return {"GT":image_tensor_resize,"inpaint_image":inpaint_tensor_resize,"inpaint_mask":mask_tensor_resize,"ref_imgs":ref_image_tensor}
   
   
    def __getitem_old__(self, index):

        
        img_path = self.imgs[index]
        img_p = Image.open(img_path).convert('RGB')
        # if self.img_transform is not None:
        #     img = self.img_transform(img)

        label = self.labels[index]
        label = Image.open(label).convert('L')
        # Assuming that 'label' is your binary mask (black and white image)
        label = np.array(label)  # Convert the label to a NumPy array if it's not already

        # Find the coordinates of the non-zero (white) pixels in the mask
        non_zero_coords = np.column_stack(np.where(label == 1))

        # Find the minimum and maximum x and y coordinates to get the bounding box
        min_x, min_y = np.min(non_zero_coords, axis=0)
        max_x, max_y = np.max(non_zero_coords, axis=0)

        # Add padding if needed
        padding = 0
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = min(img_p.size[0], max_x + padding)
        max_y = min(img_p.size[1], max_y + padding)

        # The bounding box coordinates are now (min_x, min_y, max_x, max_y)
        # Scale the bounding box coordinates to match the image size (1024x1024)
        min_x *= 2
        min_y *= 2
        max_x *= 2
        max_y *= 2
        bbox = [min_x, min_y, max_x, max_y]
        
        if self.label_transform is not None:
            label= self.label_transform(label)
 

        if self.load_vis_img:
            label_vis = self.labels_vis[index]
            label_vis = Image.open(label_vis).convert('RGB')
            label_vis = TO_TENSOR(label_vis)
        else:
            label_vis = -1  # unified interface
        
        # img_p, label, label_vis = self.load_single_image(index)
        # bbox=[30,50,60,100]
   
        ### Get reference image
        bbox_pad=copy.copy(bbox)
        bbox_pad[0]=bbox[0]-min(10,bbox[0]-0)
        bbox_pad[1]=bbox[1]-min(10,bbox[1]-0)
        bbox_pad[2]=bbox[2]+min(10,img_p.size[0]-bbox[2])
        bbox_pad[3]=bbox[3]+min(10,img_p.size[1]-bbox[3])
        img_p_np=cv2.imread(img_path)
        img_p_np = cv2.cvtColor(img_p_np, cv2.COLOR_BGR2RGB)
        ref_image_tensor=img_p_np[bbox_pad[1]:bbox_pad[3],bbox_pad[0]:bbox_pad[2],:]
        ref_image_tensor=self.random_trans(image=ref_image_tensor)
        ref_image_tensor=Image.fromarray(ref_image_tensor["image"])
        ref_image_tensor=get_tensor_clip()(ref_image_tensor)



        ### Generate mask
        image_tensor = get_tensor()(img_p)
        W,H = img_p.size

        extended_bbox=copy.copy(bbox)
        left_freespace=bbox[0]-0
        right_freespace=W-bbox[2]
        up_freespace=bbox[1]-0
        down_freespace=H-bbox[3]
        extended_bbox[0]=bbox[0]-random.randint(0,int(0.4*left_freespace))
        extended_bbox[1]=bbox[1]-random.randint(0,int(0.4*up_freespace))
        extended_bbox[2]=bbox[2]+random.randint(0,int(0.4*right_freespace))
        extended_bbox[3]=bbox[3]+random.randint(0,int(0.4*down_freespace))

        prob=random.uniform(0, 1)
        if prob<self.arbitrary_mask_percent:
            mask_img = Image.new('RGB', (W, H), (255, 255, 255)) 
            bbox_mask=copy.copy(bbox)
            extended_bbox_mask=copy.copy(extended_bbox)
            top_nodes = np.asfortranarray([
                            [bbox_mask[0],(bbox_mask[0]+bbox_mask[2])/2 , bbox_mask[2]],
                            [bbox_mask[1], extended_bbox_mask[1], bbox_mask[1]],
                        ])
            down_nodes = np.asfortranarray([
                    [bbox_mask[2],(bbox_mask[0]+bbox_mask[2])/2 , bbox_mask[0]],
                    [bbox_mask[3], extended_bbox_mask[3], bbox_mask[3]],
                ])
            left_nodes = np.asfortranarray([
                    [bbox_mask[0],extended_bbox_mask[0] , bbox_mask[0]],
                    [bbox_mask[3], (bbox_mask[1]+bbox_mask[3])/2, bbox_mask[1]],
                ])
            right_nodes = np.asfortranarray([
                    [bbox_mask[2],extended_bbox_mask[2] , bbox_mask[2]],
                    [bbox_mask[1], (bbox_mask[1]+bbox_mask[3])/2, bbox_mask[3]],
                ])
            top_curve = bezier.Curve(top_nodes,degree=2)
            right_curve = bezier.Curve(right_nodes,degree=2)
            down_curve = bezier.Curve(down_nodes,degree=2)
            left_curve = bezier.Curve(left_nodes,degree=2)
            curve_list=[top_curve,right_curve,down_curve,left_curve]
            pt_list=[]
            random_width=5
            for curve in curve_list:
                x_list=[]
                y_list=[]
                for i in range(1,19):
                    if (curve.evaluate(i*0.05)[0][0]) not in x_list and (curve.evaluate(i*0.05)[1][0] not in y_list):
                        pt_list.append((curve.evaluate(i*0.05)[0][0]+random.randint(-random_width,random_width),curve.evaluate(i*0.05)[1][0]+random.randint(-random_width,random_width)))
                        x_list.append(curve.evaluate(i*0.05)[0][0])
                        y_list.append(curve.evaluate(i*0.05)[1][0])
            mask_img_draw=ImageDraw.Draw(mask_img)
            mask_img_draw.polygon(pt_list,fill=(0,0,0))
            mask_tensor=get_tensor(normalize=False, toTensor=True)(mask_img)[0].unsqueeze(0)
        else:
            mask_img=np.zeros((H,W))
            mask_img[extended_bbox[1]:extended_bbox[3],extended_bbox[0]:extended_bbox[2]]=1
            mask_img=Image.fromarray(mask_img)
            mask_tensor=1-get_tensor(normalize=False, toTensor=True)(mask_img)

        ### Crop square image
        if W > H:
            left_most=extended_bbox[2]-H
            if left_most <0:
                left_most=0
            right_most=extended_bbox[0]+H
            if right_most > W:
                right_most=W
            right_most=right_most-H
            if right_most<= left_most:
                image_tensor_cropped=image_tensor
                mask_tensor_cropped=mask_tensor
            else:
                left_pos=random.randint(left_most,right_most) 
                free_space=min(extended_bbox[1]-0,extended_bbox[0]-left_pos,left_pos+H-extended_bbox[2],H-extended_bbox[3])
                random_free_space=random.randint(0,int(0.6*free_space))
                image_tensor_cropped=image_tensor[:,0+random_free_space:H-random_free_space,left_pos+random_free_space:left_pos+H-random_free_space]
                mask_tensor_cropped=mask_tensor[:,0+random_free_space:H-random_free_space,left_pos+random_free_space:left_pos+H-random_free_space]
        
        elif  W < H:
            upper_most=extended_bbox[3]-W
            if upper_most <0:
                upper_most=0
            lower_most=extended_bbox[1]+W
            if lower_most > H:
                lower_most=H
            lower_most=lower_most-W
            if lower_most<=upper_most:
                image_tensor_cropped=image_tensor
                mask_tensor_cropped=mask_tensor
            else:
                upper_pos=random.randint(upper_most,lower_most) 
                free_space=min(extended_bbox[1]-upper_pos,extended_bbox[0]-0,W-extended_bbox[2],upper_pos+W-extended_bbox[3])
                random_free_space=random.randint(0,int(0.6*free_space))
                image_tensor_cropped=image_tensor[:,upper_pos+random_free_space:upper_pos+W-random_free_space,random_free_space:W-random_free_space]
                mask_tensor_cropped=mask_tensor[:,upper_pos+random_free_space:upper_pos+W-random_free_space,random_free_space:W-random_free_space]
        else:
            image_tensor_cropped=image_tensor
            mask_tensor_cropped=mask_tensor

        image_tensor_resize=T.Resize([self.args['image_size'],self.args['image_size']])(image_tensor_cropped)
        mask_tensor_resize=T.Resize([self.args['image_size'],self.args['image_size']])(mask_tensor_cropped)
        inpaint_tensor_resize=image_tensor_resize*mask_tensor_resize
        
        # save_image(image_tensor_resize, "Train_data_images/"+str(index)+'_image_tensor_resize.png')
        # save_image(inpaint_tensor_resize, "Train_data_images/"+ str(index)+'_inpaint_tensor_resize.png')
        # save_image(mask_tensor_resize, "Train_data_images/"+ str(index)+'_mask_tensor_resize.png')
        # save_image(ref_image_tensor,  "Train_data_images/"+str(index)+'_ref_image_tensor.png')
        
        return {"GT":image_tensor_resize,"inpaint_image":inpaint_tensor_resize,"inpaint_mask":mask_tensor_resize,"ref_imgs":ref_image_tensor}


    def __len__(self):
        return self.length
    
    


#####DPOFaceDataset########################

    
class DPOFaceDataset(data.Dataset):
    def __init__(self, data_manifest_path, args, **kwargs):
        """
        ... (docstring 和 __init__ 的其他部分保持不变) ...
        """
        super().__init__()
        with open(data_manifest_path, "r") as f:
            self.data_list = json.load(f)

        self.args = args
        # 从 args 中读取 image_size，与 config 文件一致
        self.img_size = int(args.get("image_size", 512)) 

        # 1. 变换
        self.to_img = get_tensor(normalize=True, toTensor=True)      
        self.to_mask = get_tensor(normalize=False, toTensor=True)    
        self.to_clip = get_tensor_clip() # 即 get_tensor_clip()
        
        # 2. Resize
        self.resize_img = T.Resize([self.img_size, self.img_size], interpolation=T.InterpolationMode.BILINEAR)
        self.resize_mask = T.Resize([self.img_size, self.img_size], interpolation=T.InterpolationMode.NEAREST)

        # 3. ref_imgs 的增强
        self.random_trans = A.Compose([
            A.Resize(height=224, width=224),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=20),
            A.Blur(p=0.3),
            A.ElasticTransform(p=0.3), 
        ])

        # 4. 语义 mask 的保留类别
        if 'preserve_mask_src' in args:
            self.preserve = args['preserve_mask_src']
        elif 'preserve_mask' in args:
            self.preserve = args['preserve_mask']
        else:
            raise ValueError("Error: 'preserve_mask_src' or 'preserve_mask' not found in args.")
        
        print(f"[DPOFaceDataset] Using preserve list: {self.preserve}")
        # (确保 un_norm_clip 和 rearrange 已在文件顶部定义)
        if 'un_norm_clip' not in globals() or 'rearrange' not in globals():
            raise ImportError("Error: 'un_norm_clip' or 'rearrange' function not found.")

    def __len__(self):
        return len(self.data_list)

    def _build_inpaint_from_mask(self, mask_pil):
        # (此函数不变, 它正确地处理 Target Mask)
        mask_np = np.array(mask_pil)
        keep = np.isin(mask_np, self.preserve)
        converted = np.zeros_like(mask_np, dtype=np.uint8)
        converted[keep] = 255
        mask_keep_pil = Image.fromarray(converted).convert("L")
        mask_keep_tensor = self.to_mask(mask_keep_pil) 
        inpaint_mask = 1.0 - mask_keep_tensor
        inpaint_mask = self.resize_mask(inpaint_mask)
        scale = random.uniform(0.5, 1.0)
        inpaint_mask = decow(inpaint_mask.unsqueeze(0), scale=scale).squeeze(0)
        return inpaint_mask

    def __getitem__(self, idx):
        s = self.data_list[idx]

        # 1. 读取所有 PIL 图像
        ref_pil      = Image.open(s["path_B_source"]).convert("RGB")
        ref_mask_pil = Image.open(s["path_B_mask"]).convert("L")   # Source Mask
        tgt_mask_pil = Image.open(s["path_D_mask"]).convert("L")   # Target Mask
        
        win_pil      = Image.open(s["path_A_chosen"]).convert("RGB")
        lose_pil     = Image.open(s["path_E_rejected"]).convert("RGB")
        tgt_pil      = Image.open(s.get("path_D_target")) if s.get("path_D_target") else None
        if tgt_pil:
            tgt_pil = tgt_pil.convert("RGB")

        # -----------------------------------------------------------------
        # 2. [关键修复] 处理 GT_w, GT_l, 和 base_img
        # -----------------------------------------------------------------
        
        # A. 创建一个用于 Tensors 的 Resize 变换
        tensor_resize_op = T.Resize([self.img_size, self.img_size])

        # B. (Tensor) 先将 *完整尺寸* PIL 转为 Tensor
        GT_w_full_tensor = self.to_img(win_pil)
        GT_l_full_tensor = self.to_img(lose_pil)
        base_pil = tgt_pil if tgt_pil is not None else win_pil
        base_img_full_tensor = self.to_img(base_pil)

        # C. (Resize) 再对 Tensor 进行 Resize
        GT_w = tensor_resize_op(GT_w_full_tensor)
        GT_l = tensor_resize_op(GT_l_full_tensor)
        base_img = tensor_resize_op(base_img_full_tensor)
        
        # D. 保存未经mask处理的B和D
        # 参考 celebA.py 的 ref_imgs_nomask 处理方式：
        # - B图像(ref_img_raw): resize到224x224，使用CLIP归一化（用于CLIP特征提取）
        # - D图像(tgt_img_raw): 保持标准归一化（base_img已处理，用于训练）
        ref_pil_224 = ref_pil.resize((224, 224), Image.BILINEAR)
        ref_img_raw = self.to_clip(ref_pil_224)  # B图像（未mask，224x224，CLIP归一化）
        tgt_img_raw = base_img  # D图像（未mask，512x512，标准归一化）
        
        # -----------------------------------------------------------------
        # 3. 处理 inpaint_mask (第一次随机调用, 顺序正确)
        # -----------------------------------------------------------------
        inpaint_mask = self._build_inpaint_from_mask(tgt_mask_pil) 
        
        # -----------------------------------------------------------------
        # 4. [关键修复] 创建 inpaint_image
        #    匹配原始 CelebAdataset 的 "GT * Mask" 逻辑
        # -----------------------------------------------------------------
        inpaint_image = base_img * inpaint_mask  # <-- 修复了这一行
        
        # -----------------------------------------------------------------
        # 5. 处理 ref_imgs (第二次随机调用, 顺序和逻辑均正确)
        # -----------------------------------------------------------------
        
        # A. (使用 ref_mask_pil) 创建二值 mask (0/255 numpy)
        ref_mask_np = np.array(ref_mask_pil)
        keep_ref = np.isin(ref_mask_np, self.preserve)
        converted_ref_np = np.zeros_like(ref_mask_np, dtype=np.uint8)
        converted_ref_np[keep_ref] = 255
        
        # B. 转换为 mask_tensor (0.0/1.0 Tensor, 1=洞)
        mask_tensor_for_ref = 1.0 - self.to_mask(Image.fromarray(converted_ref_np).convert('L'))
        
        # C. 用 BILINEAR 缩放 mask_tensor 到 ref_pil 的尺寸
        resized_mask_tensor = T.Resize(ref_pil.size, interpolation=T.InterpolationMode.BILINEAR)(mask_tensor_for_ref)
        
        # D. 创建 mask_ref (0.0/1.0 Tensor, 0=洞, 1=保留)
        mask_ref = 1.0 - resized_mask_tensor
        
        # E. 【Norm 1】在 *未遮罩* 的 ref_pil 上进行第一次归一化
        tensor_clip = self.to_clip(ref_pil)
        
        # F. 【Mask】将 mask 应用于 *已归一化* 的 tensor
        tensor_clip = tensor_clip * mask_ref
        
        # G. 【Un-Norm】
        tensor_unclip = un_norm_clip(tensor_clip)
        np_unclip_255 = 255. * rearrange(tensor_unclip, 'c h w -> h w c').cpu().numpy()
        
        # H. 【不做数据增强，只 resize】⭐ 修改：去掉数据增强，保持稳定
        # 原因：数据增强（翻转、旋转、模糊）会让参考图像变化太大，影响条件编码
        ref_pil_processed = Image.fromarray(np_unclip_255.astype(np.uint8))
        ref_pil_processed = ref_pil_processed.resize((224, 224), Image.BILINEAR)
        
        # I. 【Norm 2】
        ref_imgs = self.to_clip(ref_pil_processed)
        
        # -----------------------------------------------------------------

        # 最终输出
        # 注意：GT 必须始终存在，因为 get_input 函数会使用它
        out = {
            "GT_w":          GT_w,            
            "GT_l":          GT_l,
            "GT":            base_img,        # 总是添加 GT（base_img 已经是 resized 的）
            "inpaint_image": inpaint_image,   
            "inpaint_mask":  inpaint_mask,    
            "ref_imgs":      ref_imgs,        
            # 新增：未经mask处理的B和D图像（用于SFT训练）
            "ref_img_raw":   ref_img_raw,    # B图像（Source，未mask）
            "tgt_img_raw":   tgt_img_raw,    # D图像（Target，未mask）
        }
            
        return out
########################DPOFaceDataset########################


    
  