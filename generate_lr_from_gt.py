import torch
import torch.nn.functional as F
import numpy as np
import random
from degradation import random_add_gaussian_noise_pt, random_add_poisson_noise_pt,circular_lowpass_kernel, random_mixed_kernels
from data_process import DiffJPEG, filter2D, USMSharp
import os
import random
import torch
import numpy as np
import math

from PIL import Image
import cv2



def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)

class KernelGenerator:
    def __init__(self, opt):
        self.opt = opt
        self.kernel_range = [2 * v + 1 for v in range(3, 11)]
        self.pulse_tensor = torch.zeros(21, 21).float()  # pulse tensor for no blur effect
        self.pulse_tensor[10, 10] = 1

    def generate_kernels(self):
        # 第一阶段降质的 kernel1
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt['sinc_prob']:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel1 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel1 = random_mixed_kernels(
                self.opt['kernel_list'],
                self.opt['kernel_prob'],
                kernel_size,
                self.opt['blur_sigma'],
                self.opt['blur_sigma'],
                [-math.pi, math.pi],
                self.opt['betag_range'],
                self.opt['betap_range'],
                noise_range=None
            )
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel1 = np.pad(kernel1, ((pad_size, pad_size), (pad_size, pad_size)))

        # 第二阶段降质的 kernel2
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt['sinc_prob2']:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.opt['kernel_list2'],
                self.opt['kernel_prob2'],
                kernel_size,
                self.opt['blur_sigma2'],
                self.opt['blur_sigma2'],
                [-math.pi, math.pi],
                self.opt['betag_range2'],
                self.opt['betap_range2'],
                noise_range=None
            )
          # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # 最终的 sinc kernel
        if np.random.uniform() < self.opt['final_sinc_prob']:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
        else:
            sinc_kernel = self.pulse_tensor

        # 转换为 tensor
        kernel1 = torch.FloatTensor(kernel1)
        kernel2 = torch.FloatTensor(kernel2)
        sinc_kernel = torch.FloatTensor(sinc_kernel)
        return kernel1, kernel2, sinc_kernel

# 配置参数
opt = {
    'blur_kernel_size': 21,
    'kernel_list': ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
    'kernel_prob': [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
    'sinc_prob': 0.1,
    'blur_sigma': [0.2, 3],
    'betag_range': [0.5, 4],
    'betap_range': [1, 2],
    'scale': 4,
    'blur_kernel_size2': 21,
    'kernel_list2': ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
    'kernel_prob2': [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
    'sinc_prob2': 0.1,
    'blur_sigma2': [0.2, 1.5],
    'betag_range2': [0.5, 4],
    'betap_range2': [1, 2],
    
    'final_sinc_prob': 0.8,
        # the first degradation process
    'resize_prob': [0.2, 0.7, 0.1],  # up, down, keep
    'resize_range': [0.15, 1.5],
    'gaussian_noise_prob': 0.5,
    'noise_range': [1, 30],
    'poisson_scale_range': [0.05, 3],
    'gray_noise_prob': 0.4,
    'jpeg_range': [30, 95],

    # the second degradation process
    'second_blur_prob': 0.8,
    'resize_prob2': [0.3, 0.4, 0.3],  # up, down, keep
    'resize_range2': [0.3, 1.2],
    'gaussian_noise_prob2': 0.5,
    'noise_range2': [1, 25],
    'poisson_scale_range2': [0.05, 2.5],
    'gray_noise_prob2': 0.4,
    'jpeg_range2': [30, 95],

}

# 生成降质核
kernel_generator = KernelGenerator(opt)










# 初始化JPEG压缩和USM锐化器
jpeger = DiffJPEG(differentiable=False).cuda()  # 模拟JPEG压缩伪影
usm_sharpener = USMSharp().cuda()  # do usm sharpening


def generate_lr_from_gt(gt,opt):
    """基于GT图像和降质参数生成LR图像."""
    kernel1, kernel2, sinc_kernel = kernel_generator.generate_kernels()
    ori_h, ori_w = gt.size()[2:4]
    kernel1 = kernel1.cuda()
    kernel2 = kernel2.cuda()
    sinc_kernel = sinc_kernel.cuda()
    # gt_usm =  usm_sharpener(gt.cuda())
    gt_usm=gt.cuda()
    # 第一次降质过程
    out = filter2D(gt_usm, kernel1)
    updown_type = random.choices(['up', 'down', 'keep'], opt['resize_prob'])[0]
    if updown_type == 'up':
        scale = np.random.uniform(1, opt['resize_range'][1])
    elif updown_type == 'down':
        scale = np.random.uniform(opt['resize_range'][0], 1)
    else:
        scale = 1

    mode = random.choice(['area', 'bilinear', 'bicubic'])
    out = F.interpolate(out, scale_factor=scale, mode=mode)
    gray_noise_prob = opt['gray_noise_prob']
    if np.random.uniform() < opt['gaussian_noise_prob']:
        out = random_add_gaussian_noise_pt(
            out, 
            sigma_range=opt['noise_range'], 
            clip=True, 
            rounds=False, 
            gray_prob=gray_noise_prob
    )
    else:
        out = random_add_poisson_noise_pt(
            out, 
            scale_range=opt['poisson_scale_range'], 
            gray_prob=gray_noise_prob, 
            clip=True, 
            rounds=False
        )

    
    jpeg_p = out.new_zeros(out.size(0)).uniform_(*opt['jpeg_range'])
    out = torch.clamp(out, 0, 1)
    out = jpeger(out, quality=jpeg_p) 
    # 第二次降质过程
    if np.random.uniform() < opt['second_blur_prob']:
        out = filter2D(out, kernel2)
    
    updown_type = random.choices(['up', 'down', 'keep'], opt['resize_prob2'])[0]
    if updown_type == 'up':
        scale = np.random.uniform(1, opt['resize_range2'][1])
    elif updown_type == 'down':
        scale = np.random.uniform(opt['resize_range2'][0], 1)
    else:
        scale = 1

    mode = random.choice(['area', 'bilinear', 'bicubic'])

    out = F.interpolate(out, size=(int(ori_h / opt['scale'] * scale), int(ori_w / opt['scale'] * scale)), mode=mode)
    
    gray_noise_prob = opt['gray_noise_prob2']
    if np.random.uniform() < opt['gaussian_noise_prob2']:
        out = random_add_gaussian_noise_pt(
            out,
            sigma_range=opt['noise_range2'],
            clip=True,
            rounds=False,
            gray_prob=gray_noise_prob
        )
    else:
        out = random_add_poisson_noise_pt(
            out,
            scale_range=opt['poisson_scale_range2'],
            gray_prob=gray_noise_prob,
            clip=True,
            rounds=False
        )

    
    if np.random.uniform() < 0.5:
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, size=(ori_h // opt['scale'], ori_w // opt['scale']), mode=mode)
        out = filter2D(out, sinc_kernel)
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*opt['jpeg_range2'])
        out = torch.clamp(out, 0, 1)
        out = jpeger(out, quality=jpeg_p)
    else:
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*opt['jpeg_range2'])
        out = torch.clamp(out, 0, 1)
        out = jpeger(out, quality=jpeg_p)
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, size=(ori_h // opt['scale'], ori_w // opt['scale']), mode=mode)
        out = filter2D(out, sinc_kernel)

    return torch.clamp((out * 255.0).round(), 0, 255) 




# 处理文件夹中的所有图片并保存
def process_and_save_images_in_folder(folder_path, save_folder):
    # 如果目标文件夹不存在，创建它
    os.makedirs(save_folder, exist_ok=True)
    
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            img_path = os.path.join(folder_path, filename)
            
            # 读取图片
            img = cv2.imread(img_path)
            if img is None:
                print(f"无法加载图片: {img_path}")
                continue
            
            # 转换为 tensor，并加上 batch 维度
            img_tensor = img2tensor(img/255.0).unsqueeze(0)  # 形状为 (1, C, H, W)
           
            
            # 生成降质图像 (LR 图像)
            degra_tensor = generate_lr_from_gt(img_tensor, opt)  # 此处应保证generate_lr_from_gt支持多batch
            
            # 去掉 batch 维度，并转换为 numpy 数组格式
            degra_img = degra_tensor.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)  # 形状为 (H, W, C)
         
            
            # 将降质图像从 RGB 转换为 BGR，并从 [0, 1] 转换为 [0, 255]
            degra_img = degra_img.astype(np.uint8)
            degra_img = cv2.cvtColor(degra_img, cv2.COLOR_RGB2BGR)
            
            # 保存图像
            save_path = os.path.join(save_folder, filename)
            cv2.imwrite(save_path, degra_img)
            print(f"保存降质图像: {save_path}")

# 使用示例
folder_path = 'D:\\SR\\DRealSR\\test_LR\\2xLR'  # 原始图片文件夹路径
save_folder = 'D:\\SR\\DRealSR\\test_LR\\8xLR_degra'  # 降质后图片保存文件夹路径
process_and_save_images_in_folder(folder_path, save_folder)








