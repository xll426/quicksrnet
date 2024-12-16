import os
import random
import torch
import numpy as np
import math
from degradation import circular_lowpass_kernel, random_mixed_kernels

from PIL import Image
import cv2

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
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

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
    
    'blur_kernel_size2': 21,
    'kernel_list2': ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
    'kernel_prob2': [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
    'sinc_prob2': 0.1,
    'blur_sigma2': [0.2, 1.5],
    'betag_range2': [0.5, 4],
    'betap_range2': [1, 2],
    
    'final_sinc_prob': 0.8
}

# 生成降质核
kernel_generator = KernelGenerator(opt)
kernel1, kernel2, sinc_kernel = kernel_generator.generate_kernels()
print("Kernel1:", kernel1)
print("Kernel2:", kernel2)
print("Sinc Kernel:", sinc_kernel)
