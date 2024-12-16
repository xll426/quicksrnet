import os
import torch
from torch.utils import data as data
import cv2
import numpy as np
import random
from torchvision.transforms.functional import normalize

# 从 bytes 读取图像
def imfrombytes(content, flag='color', float32=False):
    """Read an image from bytes."""
    img_np = np.frombuffer(content, np.uint8)
    imread_flags = {'color': cv2.IMREAD_COLOR, 'grayscale': cv2.IMREAD_GRAYSCALE, 'unchanged': cv2.IMREAD_UNCHANGED}
    img = cv2.imdecode(img_np, imread_flags[flag])
    if float32:
        img = img.astype(np.float32) / 255.
    return img



def yfrombytes(content, flag='color', float32=False):
    """Read an image from bytes."""
    img_np = np.frombuffer(content, np.uint8)
    imread_flags = {'color': cv2.IMREAD_COLOR, 'grayscale': cv2.IMREAD_GRAYSCALE, 'unchanged': cv2.IMREAD_UNCHANGED}
    img = cv2.imdecode(img_np, imread_flags[flag])
  
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # 提取 Y 分量
    y_component = img_yuv[:, :, 0]  # Y 分量是 YUV 图像的第一个通道

    # 如果需要，将 Y 分量转换为浮点数并归一化到 [0, 1] 范围
    if float32:
        y_component = y_component.astype(np.float32) / 255.0
        
    return y_component
   

# 图像增强函数
def augment(imgs, hflip=True, rotation=True):
    """Augment: horizontal flips OR rotate."""
    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        if hflip:  # horizontal flip
            img = cv2.flip(img, 1)
        if vflip:  # vertical flip
            img = cv2.flip(img, 0)
        # if rot90:  # rotate 90 degrees
        #     img = np.transpose(img, (1, 0, 2))
        return img

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]
    return imgs

# 随机裁剪函数
def paired_random_crop(img_gts, img_lqs, gt_patch_size, scale):
    """Paired random crop."""
    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]

    h_lq, w_lq = img_lqs[0].shape[0:2]
    lq_patch_size = gt_patch_size // scale

    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    img_lqs = [v[top:top + lq_patch_size, left:left + lq_patch_size, ...] for v in img_lqs]
    top_gt, left_gt = int(top * scale), int(left * scale)
    img_gts = [v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...] for v in img_gts]

    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lqs) == 1:
        img_lqs = img_lqs[0]
    return img_gts, img_lqs

# Numpy array 转换为 Tensor
def img2tensor(imgs, bgr2rgb=True, float32=True):
    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)

# Numpy array 转换为 Tensor
def y2tensor(imgs,float32=True):
    def _totensor(img,  float32):
        
        img = torch.from_numpy(np.expand_dims(img, axis=0))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, float32) for img in imgs]
    else:
        return _totensor(imgs, float32)


class SuperResolutionPairedDataset(data.Dataset):
    """Paired image dataset for super-resolution."""
    def __init__(self, opt):
        super(SuperResolutionPairedDataset, self).__init__()
        self.opt = opt
        self.gt_folder = opt['dataroot_gt']
        self.lq_folder = opt['dataroot_lq']
        # self.mean = opt.get('mean', None)
        # self.std = opt.get('std', None)
        self.phase = opt['phase']
        
        # 加载图像路径
     
        meta_info_file = opt['meta_info']
        with open(meta_info_file, 'r') as f:
            self.paths = [{'gt_path': os.path.join(self.gt_folder, line.split(', ')[0]),
                           'lq_path': os.path.join(self.lq_folder, line.split(', ')[1].strip())}
                          for line in f]

    def __getitem__(self, index):
        # 获取图像路径
        gt_path = self.paths[index]['gt_path']
        lq_path = self.paths[index]['lq_path']
        
        # 从文件读取图像
        with open(gt_path, 'rb') as f_gt, open(lq_path, 'rb') as f_lq:
            img_gt = yfrombytes(f_gt.read(), float32=True)
            img_lq = yfrombytes(f_lq.read(), float32=True)
        
        # 数据增强
        if self.phase == 'train':
            gt_size = self.opt['gt_size']
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, self.opt['scale'])
            img_gt, img_lq = augment([img_gt, img_lq], self.opt.get('use_hflip', True), self.opt.get('use_rot', True))
        
        # 转换为 PyTorch 张量并归一化
        img_gt, img_lq = y2tensor([img_gt, img_lq], float32=True)
        # if self.mean is not None or self.std is not None:
        #     normalize(img_lq, self.mean, self.std, inplace=True)
        #     normalize(img_gt, self.mean, self.std, inplace=True)
        
        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)
