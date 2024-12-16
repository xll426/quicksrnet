# import os
# import cv2
# import numpy as np
# from skimage.metrics import peak_signal_noise_ratio as psnr
# from skimage.metrics import structural_similarity as ssim

# # 文件夹路径配置
# gt_folder = 'D:\\SR\\DRealSR\\test_HR\\'  # 原始高质量图像的文件夹路径
# sr_folder = 'D:\\SR\\quicksr\\x2\\checkpoints_large\\best_quicksrnet_save\\'  # 超分辨率生成的图像文件夹路径

# # 初始化计数器和累加器
# psnr_total = 0
# ssim_total = 0
# count = 0

# # 遍历 gt 文件夹中的所有图像文件
# for filename in os.listdir(gt_folder):
#     gt_path = os.path.join(gt_folder, filename)
#     # sr_path = os.path.join(sr_folder, filename)
#     sr_path = os.path.join(sr_folder, filename[:-5]+"1.png")

#     # 检查 SR 文件夹中是否存在对应的图像
#     if not os.path.exists(sr_path):
#         print(f"Warning: {filename} not found in {sr_folder}")
#         continue

#     # 读取图像
#     gt_img = cv2.imread(gt_path)
#     sr_img = cv2.imread(sr_path)
#     gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
#     sr_img = cv2.cvtColor(sr_img, cv2.COLOR_BGR2RGB)


#     # 检查图像是否存在且大小相同
#     if gt_img is None or sr_img is None:
#         print(f"Warning: Unable to read {filename} in one of the folders")
#         continue
#     if gt_img.shape != sr_img.shape:
#         print(f"Warning: Shape mismatch for {filename}")
#         continue

#     # 计算 PSNR 和 SSIM
#     current_psnr = psnr(gt_img, sr_img, data_range=255)
#     current_ssim = ssim(gt_img, sr_img, channel_axis=2, data_range=255)

#     # 累加 PSNR 和 SSIM 值
#     psnr_total += current_psnr
#     ssim_total += current_ssim
#     count += 1

#     print(f"{filename}: PSNR={current_psnr:.2f}, SSIM={current_ssim:.4f}")

# # 计算平均 PSNR 和 SSIM
# if count > 0:
#     avg_psnr = psnr_total / count
#     avg_ssim = ssim_total / count
#     print(f"\nAverage PSNR: {avg_psnr:.2f}")
#     print(f"Average SSIM: {avg_ssim:.4f}")
# else:
#     print("No images were processed.")


import os
import cv2
import torch
import numpy as np
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

# 文件夹路径配置
gt_folder = 'D:\\SR\\DRealSR\\test_HR\\'  # 原始高质量图像的文件夹路径
sr_folder = 'D:\\SR\\quicksr\\x2\\checkpoints_large\\best_quicksrnet_save\\'  # 超分辨率生成的图像文件夹路径

# 初始化 PSNR 和 SSIM 计算器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
psnr_calculator = PeakSignalNoiseRatio(data_range=255.0).to(device)
ssim_calculator = StructuralSimilarityIndexMeasure(data_range=255.0).to(device)

# 初始化计数器和累加器
psnr_total = 0
ssim_total = 0
count = 0

# 遍历 gt 文件夹中的所有图像文件
for filename in os.listdir(gt_folder):
    gt_path = os.path.join(gt_folder, filename)
    # sr_path = os.path.join(sr_folder, filename)
    sr_path = os.path.join(sr_folder, filename[:-5] + "1.png")

    # 检查 SR 文件夹中是否存在对应的图像
    if not os.path.exists(sr_path):
        print(f"Warning: {filename} not found in {sr_folder}")
        continue

    # 读取图像并转换为RGB格式
    gt_img = cv2.imread(gt_path)
    sr_img = cv2.imread(sr_path)
    gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
    sr_img = cv2.cvtColor(sr_img, cv2.COLOR_BGR2RGB)

    # 检查图像是否存在且大小相同
    if gt_img is None or sr_img is None:
        print(f"Warning: Unable to read {filename} in one of the folders")
        continue
    if gt_img.shape != sr_img.shape:
        print(f"Warning: Shape mismatch for {filename}")
        continue

    # 转换图像为 torch 张量并将数据转移到 GPU
    gt_img_tensor = torch.from_numpy(gt_img).permute(2, 0, 1).unsqueeze(0).float().to(device)
    sr_img_tensor = torch.from_numpy(sr_img).permute(2, 0, 1).unsqueeze(0).float().to(device)


    # 获取 sr_images 和 hr_images 的公共区域
    _, _, h_sr, w_sr = sr_img_tensor.shape
    _, _, h_hr, w_hr = gt_img_tensor.shape

    # 找到公共区域的尺寸（取最小的宽和高）
    common_height = min(h_sr, h_hr)
    common_width = min(w_sr, w_hr)

    # 提取公共区域
    sr_common = sr_img_tensor[:, :, :common_height, :common_width]
    hr_common = gt_img_tensor[:, :, :common_height, :common_width]

    # 计算 PSNR 和 SSIM
    current_psnr = psnr_calculator(sr_common, hr_common).item()
    current_ssim = ssim_calculator(sr_common, hr_common).item()




    # # 计算 PSNR 和 SSIM
    # current_psnr = psnr_calculator(sr_img_tensor, gt_img_tensor).item()
    # current_ssim = ssim_calculator(sr_img_tensor, gt_img_tensor).item()

    # 累加 PSNR 和 SSIM 值
    psnr_total += current_psnr
    ssim_total += current_ssim
    count += 1

    print(f"{filename}: PSNR={current_psnr:.2f}, SSIM={current_ssim:.4f}")

# 计算平均 PSNR 和 SSIM
if count > 0:
    avg_psnr = psnr_total / count
    avg_ssim = ssim_total / count
    print(f"\nAverage PSNR: {avg_psnr:.2f}")
    print(f"Average SSIM: {avg_ssim:.4f}")
else:
    print("No images were processed.")

