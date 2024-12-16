import os
import cv2
import torch
import numpy as np
from model import QuickSRNetSmall,QuickSRNetMedium,QuickSRNetLarge # 假设你定义了这个模型

# 配置选项
opt = {
    'scale': 2,  # 超分辨率缩放因子
    'model_checkpoint': 'D:\\SR\\quicksr\\x2\\checkpoints_large\\best_quicksrnet.pth',  # 模型检查点路径
    'input_folder': 'D:\\SR\\DRealSR\\test_LR\\2xLR\\',  # 输入图像文件夹路径
    'output_folder': 'D:\\SR\\quicksr\\x2\\checkpoints_large\\best_quicksrnet_save_patch\\',  # 输出图像文件夹路径
    'patch_size': 460,  # 裁剪块大小
    'overlap': 20,  # 冗余重叠部分 (50%)
}

# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = QuickSRNetLarge(scaling_factor=opt['scale']).to(device)

# 加载预训练模型权重
checkpoint = torch.load(opt['model_checkpoint'], map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # 进入评估模式，不更新梯度

# 确保输出文件夹存在
os.makedirs(opt['output_folder'], exist_ok=True)

def predict_patch(model, patch):
    """对单个图像块进行超分辨率预测"""
    patch_tensor = torch.from_numpy(patch.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        sr_patch_tensor = model(patch_tensor)
    return sr_patch_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)

def process_large_image_in_parts(image, scale, patch_size, overlap):
    """对大图进行重叠裁剪、逐块预测并拼接"""
    h, w, c = image.shape
    stride = patch_size - overlap

    # 计算需要多少行和列来覆盖整个图像
    rows = (h - overlap) // stride + 1
    cols = (w - overlap) // stride + 1

    # 初始化空的输出图像
    sr_image = np.zeros((h * scale, w * scale, c), dtype=np.float32)

    for i in range(rows):
        for j in range(cols):
            # 计算当前块的左上角坐标
            y_start, x_start = i * stride, j * stride
            y_end = min(y_start + patch_size, h)
            x_end = min(x_start + patch_size, w)
            y_start = max(0, y_end - patch_size)
            x_start = max(0, x_end - patch_size)

            # 提取块并进行超分辨率预测
            patch = image[y_start:y_end, x_start:x_end]
            sr_patch = predict_patch(model, patch)

            # 确定高分辨率图像中的放置位置
            y_sr_start, y_sr_end = y_start * scale, y_end * scale
            x_sr_start, x_sr_end = x_start * scale, x_end * scale

            # 将超分辨率块直接覆盖到输出图像中
            sr_image[y_sr_start:y_sr_end, x_sr_start:x_sr_end] = sr_patch

    return sr_image

# 主函数
if __name__ == '__main__':
    # 遍历输入文件夹中的所有图像文件
    for img_name in os.listdir(opt['input_folder']):
        img_path = os.path.join(opt['input_folder'], img_name)
        if not os.path.isfile(img_path):
            continue

        # 读取图像
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # 对图像进行逐块预测
        sr_image = process_large_image_in_parts(image, opt['scale'], opt['patch_size'], opt['overlap'])

        # 保存结果
        sr_image = (sr_image * 255).astype(np.uint8)
        sr_image_bgr = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
        output_path = os.path.join(opt['output_folder'], os.path.splitext(img_name)[0][:-1] + '2.png')
        cv2.imwrite(output_path, sr_image_bgr)
        print(f"Saved super-resolution image to {output_path}")
