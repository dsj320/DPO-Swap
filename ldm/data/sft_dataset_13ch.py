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
    
    

#####SFTFaceDataset########################



import os
import json
import cv2
import torch
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.transforms as T
from torch.utils import data
import albumentations as A
from einops import rearrange


try:
    import models.third_party.model_resnet_d3dfr as model_resnet_d3dfr
    import models.third_party.d3dfr.bfm as bfm
except ImportError:
    print("Warning: D3DFR modules not found. Ensure 'third_party' is in python path.")    


def draw_landmarks_on_black(size, landmarks, radius=4):
    """
    在全黑背景上绘制彩虹色关键点
    Args:
        size: 图片边长 (int)
        landmarks: (68, 2) numpy array
        radius: 点的半径
    """
    # 创建黑色底图 (H, W, 3)
    img_draw = np.zeros((size, size, 3), dtype=np.uint8)
    
    # 彩虹色生成
    colors = plt.get_cmap('rainbow')(np.linspace(0, 1, len(landmarks)))
    colors = (255 * colors).astype(int)[:, :3].tolist()
    
    for i, (x, y) in enumerate(landmarks):
        # 边界检查
        if x < 0 or x >= size or y < 0 or y >= size:
            continue
            
        color = colors[i]
        #以此颜色填充圆: OpenCV使用BGR，我们这里生成的是RGB，稍后统一转
        # plt.get_cmap 生成的是 RGBA, 我们取 RGB. cv2.circle 需要 color 是 int tuple
        cv2.circle(img_draw, (int(x), int(y)), radius=radius, 
                  color=(color[0], color[1], color[2]), thickness=-1) # 使用RGB顺序，因为后续转PIL
    
    return img_draw


class SFTFaceDataset(data.Dataset):
    def __init__(self, data_manifest_path, base_3d_path,args, **kwargs):
        super().__init__()
        with open(data_manifest_path, "r") as f:
            self.data_list = json.load(f)
        self.args = args
        # 从 args 中读取 image_size，与 config 文件一致
        self.img_size = int(args.get("image_size", 512)) 

        # 基础变换
        self.to_img = get_tensor(normalize=True, toTensor=True)      
        self.to_mask = get_tensor(normalize=False, toTensor=True)    
        self.to_clip = get_tensor_clip() # 即 get_tensor_clip()
        self.resize_img = T.Resize([self.img_size, self.img_size], interpolation=T.InterpolationMode.BILINEAR)
        self.resize_mask = T.Resize([self.img_size, self.img_size], interpolation=T.InterpolationMode.NEAREST)

        # ref_imgs 的增强
        self.random_trans = A.Compose([
            A.Resize(height=224, width=224),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=20),
            A.Blur(p=0.2),
        ])

        # 语义 mask 的保留类别 - 支持根据数据类型（sft/recon）动态选择
        # 模式1: 兼容模式（旧配置）
        if 'preserve_mask' in args:
            self.preserve_src_sft = args['preserve_mask']
            self.remove_tar_sft = args['preserve_mask']
            self.preserve_src_recon = args['preserve_mask']
            self.remove_tar_recon = args['preserve_mask']
            self.dynamic_mask = False
            print(f"[SFTFaceDataset] Using legacy mode with single mask config: {self.preserve_src_sft}")
        
        # 模式2: 统一配置（sft和recon使用相同配置）
        elif 'preserve_mask_src' in args and 'remove_mask_tar' in args:
            self.preserve_src_sft = args['preserve_mask_src']
            self.remove_tar_sft = args['remove_mask_tar']
            self.preserve_src_recon = args['preserve_mask_src']
            self.remove_tar_recon = args['remove_mask_tar']
            self.dynamic_mask = False
            print(f"[SFTFaceDataset] Using unified mode - Source: {self.preserve_src_sft}, Target: {self.remove_tar_sft}")
        
        # 模式3: 动态配置（sft和recon使用不同配置）⭐ 新增
        elif 'preserve_mask_src_sft' in args:
            self.preserve_src_sft = args['preserve_mask_src_sft']
            self.remove_tar_sft = args['remove_mask_tar_sft']
            self.preserve_src_recon = args['preserve_mask_src_recon']
            self.remove_tar_recon = args['remove_mask_tar_recon']
            self.dynamic_mask = True
            print(f"[SFTFaceDataset] Using dynamic mode:")
            print(f"  - SFT    -> Source: {self.preserve_src_sft}, Target: {self.remove_tar_sft}")
            print(f"  - Recon  -> Source: {self.preserve_src_recon}, Target: {self.remove_tar_recon}")
        else:
            raise ValueError("Error: Must provide mask configurations. See comments for supported modes.")

        if 'ref_imgs_augmentation' in args:
            self.ref_imgs_augmentation = args['ref_imgs_augmentation']
        else:
            self.ref_imgs_augmentation = False
        if 'un_norm_clip' not in globals() or 'rearrange' not in globals():
            raise ImportError("Error: 'un_norm_clip' or 'rearrange' function not found.")

        print(">>> [SFTFaceDataset] Loading 3D models (D3DFR & BFM)...")
        self.device_3d = 'cpu' 
        # D3DFR 需要的输入预处理
        self.d3dfr_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5)
        ])

        # 加载 D3DFR
        d3dfr_path = os.path.join(base_3d_path, 'd3dfr_res50_nofc.pth')
        self.net_d3dfr = model_resnet_d3dfr.getd3dfr_res50(d3dfr_path).eval().to(self.device_3d)
        
        # 加载 BFM
        bfm_path = os.path.join(base_3d_path, 'BFM_model_front.mat')
        self.bfm_facemodel = bfm.BFM(
            focal=1015*256/224,     
            image_size=256, 
            bfm_model_path=bfm_path
        ).to(self.device_3d)
        print(">>> [SFTFaceDataset] 3D Models loaded.")

    def __len__(self):
        return len(self.data_list)

    def get_3d_mixed_landmark_map(self, src_pil, tgt_pil):
        """
         执行 3D 提取、混合参数、生成关键点并在黑色底图上绘制
        """
        src_tensor = self.d3dfr_transform(src_pil).unsqueeze(0).to(self.device_3d)
        tgt_tensor = self.d3dfr_transform(tgt_pil).unsqueeze(0).to(self.device_3d)
        with torch.no_grad():
            # 提取系数
            src_coeff = self.net_d3dfr(src_tensor) # (1, 257)
            tgt_coeff = self.net_d3dfr(tgt_tensor) # (1, 257)

            # 混合系数: Source ID (0:80) + Target Exp/Texture (80:)
            mixed_coeff = tgt_coeff.clone()
            mixed_coeff[:, 0:80] = src_coeff[:, 0:80]

            # 生成 68 个关键点 (基于 256x256 空间)
            # get_lm68 返回 (B, 68, 2)
            mixed_pts68_network = self.bfm_facemodel.get_lm68(mixed_coeff)[0].cpu().numpy()

        # 坐标映射: 从 256 映射到 self.img_size (例如 512)
        scale = self.img_size / 256.0
        mixed_pts_final = mixed_pts68_network * scale

        # 在黑色底图上绘制
        radius = max(2, int(self.img_size / 128)) 
        black_bg_img = draw_landmarks_on_black(self.img_size, mixed_pts_final, radius=radius)
        return Image.fromarray(black_bg_img)

    def _build_inpaint_from_mask(self, mask_pil, preserve_list=None, return_before_augment=False):
            
        mask_np = np.array(mask_pil)
        keep = np.isin(mask_np, preserve_list)
        converted = np.zeros_like(mask_np, dtype=np.uint8)
        converted[keep] = 255
        mask_keep_pil = Image.fromarray(converted).convert("L")
        mask_keep_tensor = self.to_mask(mask_keep_pil) 
        inpaint_mask = 1.0 - mask_keep_tensor
        inpaint_mask = self.resize_mask(inpaint_mask)
        
        # 保存增强前的 mask
        inpaint_mask_before = inpaint_mask.clone()
        
        # 应用 TPS 形变增强
        scale = random.uniform(0.5, 1.0)
        inpaint_mask = decow(inpaint_mask.unsqueeze(0), scale=scale).squeeze(0)
        
        if return_before_augment:
            return inpaint_mask, inpaint_mask_before
        return inpaint_mask

    def __getitem__(self, idx):
        s = self.data_list[idx]

        label = s.get("label", "sft")  # 默认为'sft'
        
        # ⭐ 根据label动态选择mask配置
        if label == 'recon':
            preserve_src = self.preserve_src_recon
            remove_tar = self.remove_tar_recon
        else:  # 'sft'
            preserve_src = self.preserve_src_sft
            remove_tar = self.remove_tar_sft

        # 1. 读取所有 PIL 图像
        ref_pil      = Image.open(s["path_B_source"]).convert("RGB")
        ref_mask_pil = Image.open(s["path_B_mask"]).convert("L")   # Source Mask
        tgt_mask_pil = Image.open(s["path_D_mask"]).convert("L")   # Target Mask
        
        win_pil      = Image.open(s["path_A_chosen"]).convert("RGB")
        tgt_pil      = Image.open(s.get("path_D_target")) if s.get("path_D_target") else None
        if tgt_pil:
            tgt_pil = tgt_pil.convert("RGB")

        
        # A. 创建一个用于 Tensors 的 Resize 变换
        tensor_resize_op = T.Resize([self.img_size, self.img_size])

        # B. (Tensor) 先将 *完整尺寸* PIL 转为 Tensor
        GT_w_full_tensor = self.to_img(win_pil)
        base_pil = tgt_pil if tgt_pil is not None else win_pil
        base_img_full_tensor = self.to_img(base_pil)

        # C. (Resize) 再对 Tensor 进行 Resize
        GT_w = tensor_resize_op(GT_w_full_tensor)
        base_img = tensor_resize_op(base_img_full_tensor)
        
        ref_pil_224 = ref_pil.resize((224, 224), Image.BILINEAR)
        ref_img_raw = self.to_clip(ref_pil_224)  # B图像（未mask，224x224，CLIP归一化）
        tgt_img_raw = base_img  # D图像（未mask，512x512，标准归一化）
        
        # 获取增强前后的mask - 使用动态选择的 remove_tar
        inpaint_mask_augmented, inpaint_mask_before_augment = self._build_inpaint_from_mask(
            tgt_mask_pil, 
            preserve_list=remove_tar,  # ⭐ 使用动态选择的配置
            return_before_augment=True
        ) 
        
        # 根据label决定使用哪个mask
        # 如果是recon任务，使用增强后的mask（增加数据多样性）
        # 如果是sft任务，使用未增强的mask（保持精确对应）
        if label == 'recon':
            inpaint_mask = inpaint_mask_augmented
        else:
            inpaint_mask = inpaint_mask_before_augment
        
        inpaint_image = base_img * inpaint_mask
        
        
        # A. (使用 ref_mask_pil) 创建二值 mask (0/255 numpy)
        # 使用 preserve_src：保留核心人脸区域，用于身份特征提取
        ref_mask_np = np.array(ref_mask_pil)
        keep_ref = np.isin(ref_mask_np, preserve_src)  # ⭐ 使用动态选择的配置
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
        

        # === 👇 核心修改开始 👇 ===
        
        # 1. 确保转成 uint8 格式 (Albumentations 需要这个格式)
        np_img_face = np_unclip_255.astype(np.uint8)

        # 2. 执行 Ref 增强 (如果开关打开)
        if self.ref_imgs_augmentation:
            # 直接把图片扔进去变换
            # 因为背景已经被 Mask 变成了黑色(0)，旋转时填黑边是安全的
            augmented = self.random_trans(image=np_img_face)
            np_img_face = augmented['image']

        # 3. 转回 PIL 并 Resize 到 224 (CLIP 的输入尺寸)
        # 此时的 np_img_face 已经是增强过（比如翻转过）的了
        ref_pil_processed = Image.fromarray(np_img_face)
        ref_pil_processed = ref_pil_processed.resize((224, 224), Image.BILINEAR)
        
        # 4. 【Norm 2】最终归一化到CLIP格式
        ref_imgs = self.to_clip(ref_pil_processed)
        
        # === 👆 核心修改结束 👆 ===


        # -----------------------------------------------------------------
        # 3. [新增] 生成 3D Landmark Guide Map
        # -----------------------------------------------------------------
        # 输入: Source (ref_pil) 和 Target (base_pil)
        # 输出: 黑色背景 + 混合后的彩虹色关键点 PIL
        mixed_lm_pil = self.get_3d_mixed_landmark_map(ref_pil, base_pil)
        
        # 转为 Tensor 并归一化到 [-1, 1] (与 GT, inpaint_image 格式一致)
        mixed_3d_landmarks = self.to_img(mixed_lm_pil)
        
        
        out = {
            # ─────────────────────────────────────────────────────────────
            # 1. GT_w 真正的监督值
            # ─────────────────────────────────────────────────────────────
            "GT_w": GT_w,
            # 来源: path_A_chosen (A图像)
            # 内容: 好的修复结果（换脸后的正确结果）
            # 形状: (3, 512, 512)
            # 归一化: 标准归一化 [-1, 1], mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)
            # mask状态: 未mask（完整图像）
            
            # ─────────────────────────────────────────────────────────────
            # GT target,自监督当中也是gt,这里保留名字了
            # ─────────────────────────────────────────────────────────────
            "GT": base_img,
            # 来源: path_D_target
            # 内容: 目标图像（提供姿态和背景）
            # 形状: (3, 512, 512)
            # 归一化: 标准归一化 [-1, 1]
            # mask状态: 未mask（完整图像）
            
            # ─────────────────────────────────────────────────────────────
            # inpaint_image (模型的主要输入)
            # ─────────────────────────────────────────────────────────────
            "inpaint_image": inpaint_image,
            # 来源: base_img * inpaint_mask
            # 内容: target图像被mask遮罩后的结果（脸部被遮罩变黑，背景保留），如果是recon任务，则使用增强后的mask，如果是sft任务，则使用未增强的mask
            # 形状: (3, 512, 512)
            # 归一化: 标准归一化 [-1, 1]
            # mask状态: 已mask（脸部区域值为0，背景区域保留原值）
            # 具体: inpaint_mask中 1.0的地方保留原图，0.0的地方变为0（黑色）

            # ─────────────────────────────────────────────────────────────
            # inpaint_mask ，如果是recon任务，则使用增强后的mask，如果是sft任务，则使用未增强的mask
            # ─────────────────────────────────────────────────────────────
            "inpaint_mask": inpaint_mask,
            # 来源: _build_inpaint_from_mask(增强前版本)
            # 内容: 二值mask（规则边界，未经TPS形变）
            # 形状: (1, 512, 512)
            # 数值范围: [0.0, 1.0]
            # mask语义: 1.0 = 保留区域（背景），0.0 = 遮罩区域（需要修复的脸部）
            
        
            # ─────────────────────────────────────────────────────────────
            # ref_imgs (参考图像 - 带mask，用于条件编码)
            # ─────────────────────────────────────────────────────────────
            "ref_imgs": ref_imgs,
            # 来源: source图像经过复杂的mask+归一化处理
            # 内容: source图像只保留核心人脸区域（skin,nose,eyes,brows,mouth,lips）
            #       非核心区域（ears,neck）被mask掉（值为0）
            # 形状: (3, 224, 224)
            # 归一化: CLIP归一化
            # 数值范围: 典型 [-2.0, 2.0]
            # mask状态: 已mask（只保留核心人脸，非核心区域为0）
            
            # ─────────────────────────────────────────────────────────────
            # 9. ref_img_raw (参考图像 - 未mask)
            # ─────────────────────────────────────────────────────────────
            "ref_img_raw": ref_img_raw,
            # 来源: B源图像直接resize和归一化
            # 内容: B的完整图像（包括所有区域：人脸+耳朵+脖子+背景）
            # 形状: (3, 224, 224)
            # 归一化: CLIP归一化
            # 数值范围: 典型 [-2.0, 2.0]
            # mask状态: 未mask（完整图像）
            
     

            # ─────────────────────────────────────────────────────────────
            # mixed_3d_landmarks (3D 关键点混合引导图) 
            # ─────────────────────────────────────────────────────────────
            "mixed_3d_landmarks": mixed_3d_landmarks
            # 来源: D3DFR模型提取 source ID + target Pose 混合后生成的关键点
            # 内容: 黑色背景上绘制的彩虹色关键点
            # 形状: (3, 512, 512)
            # 归一化: 标准归一化 [-1, 1]
            # 作用: 作为强几何引导 (Geometric Guidance)，辅助模型对齐五官位置
        }
            
        return out
########################DPOFaceDataset########################


    
  