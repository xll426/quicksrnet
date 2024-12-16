import os
import cv2
import torch
import numpy as np
from model import QuickSRNetSmall,QuickSRNetMedium,QuickSRNetLarge # 假设你定义了这个模型
import re

# 配置选项
opt = {
    'scale': 2,  # 超分辨率缩放因子
    'model_checkpoint': 'D:\\SR\\quicksr\\x2\\checkpoints_large\\best_quicksrnet.pth',  # 模型检查点路径
    'input_folder': 'D:\\SR\\quicksr\\cfyt\\',  # 输入图像文件夹路径
    'output_folder': 'D:\\SR\\quicksr\\cfyt_result\\',  # 输出图像文件夹路径
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

def parse_resolution_from_filename(filename):
    """
    从文件名中解析宽度和高度
    文件名格式示例：1_3840x2160.nv12.yuv
    """
    match = re.search(r"(\d+)x(\d+)", filename)  # 匹配类似 "3840x2160" 的格式
    if match:
        width, height = map(int, match.groups())
        return width, height
    else:
        raise ValueError(f"无法从文件名中解析分辨率: {filename}")
    

def nv12_to_rgb(file_path, width, height):
    """
    从 NV12 文件解析并转换为 RGB 图像
    :param file_path: NV12 文件路径
    :param width: 图像宽度
    :param height: 图像高度
    :return: RGB 图像 (np.ndarray)
    """
    # 读取 NV12 文件
    with open(file_path, 'rb') as f:
        yuv_data = np.frombuffer(f.read(), dtype=np.uint8)

    # 解析 Y 平面
    y_plane = yuv_data[:width * height].reshape((height, width))

    # 解析 UV 平面（交错存储，UV 宽高是 Y 的一半）
    uv_plane = yuv_data[width * height:].reshape((height // 2, width))
    u_plane = uv_plane[:, 0::2]  # 偶数列是 U 分量
    v_plane = uv_plane[:, 1::2]  # 奇数列是 V 分量

    # 将 UV 分量上采样到与 Y 分量相同大小（线性插值）
    u_plane_upsampled = np.repeat(np.repeat(u_plane, 2, axis=0), 2, axis=1)
    v_plane_upsampled = np.repeat(np.repeat(v_plane, 2, axis=0), 2, axis=1)

    # 转换为浮点类型，方便颜色空间转换计算
    y_plane = y_plane.astype(np.float32)
    u_plane_upsampled = u_plane_upsampled.astype(np.float32) - 128.0
    v_plane_upsampled = v_plane_upsampled.astype(np.float32) - 128.0

    # YUV 转换为 RGB
    r = y_plane + 1.402 * v_plane_upsampled
    g = y_plane - 0.344136 * u_plane_upsampled - 0.714136 * v_plane_upsampled
    b = y_plane + 1.772 * u_plane_upsampled

    # 将 RGB 分量裁剪到 [0, 255] 范围并转换为 uint8
    r = np.clip(r, 0, 255).astype(np.uint8)
    g = np.clip(g, 0, 255).astype(np.uint8)
    b = np.clip(b, 0, 255).astype(np.uint8)

    # 合并 RGB 分量为最终图像
    rgb_image = np.stack((r, g, b), axis=-1)
    return rgb_image.astype(np.float32) / 255.0 

def read_nv12(file_path, width, height):
    """从NV12文件读取并转换为RGB图像"""
    with open(file_path, 'rb') as f:
        yuv_data = np.frombuffer(f.read(), dtype=np.uint8)
    
    y_size = width * height
    uv_size = y_size // 2

    # 提取Y平面和UV平面
    y_plane = yuv_data[:y_size].reshape((height, width))
    uv_plane = yuv_data[y_size:y_size + uv_size].reshape((height // 2, width))

    # 从UV平面分离U和V分量
    u_plane = uv_plane[:, 0::2]  # 偶数列是U
    v_plane = uv_plane[:, 1::2]  # 奇数列是V

    # 将UV分量上采样到Y分量大小
    u_up = cv2.resize(u_plane, (width, height), interpolation=cv2.INTER_LINEAR)
    v_up = cv2.resize(v_plane, (width, height), interpolation=cv2.INTER_LINEAR)

    # 合并YUV分量
    yuv = np.stack((y_plane, u_up, v_up), axis=-1).astype(np.float32)

    # YUV转换为RGB
    rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
    return rgb / 255.0


def write_nv12(file_path, rgb_image):
    """将RGB图像转换为NV12格式并保存"""
    height, width, _ = rgb_image.shape

    # RGB转换为YUV
    yuv = cv2.cvtColor((rgb_image * 255).astype(np.uint8), cv2.COLOR_RGB2YUV)
    y, u, v = cv2.split(yuv)

    # 下采样UV分量
    u_down = cv2.resize(u, (width // 2, height // 2), interpolation=cv2.INTER_LINEAR)
    v_down = cv2.resize(v, (width // 2, height // 2), interpolation=cv2.INTER_LINEAR)

    # 交错存储UV
    uv_interleaved = np.zeros((height // 2, width), dtype=np.uint8)
    uv_interleaved[:, 0::2] = u_down
    uv_interleaved[:, 1::2] = v_down

    # 保存NV12数据
    with open(file_path, 'wb') as f:
        f.write(y.tobytes())
        f.write(uv_interleaved.tobytes())

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
    for yuv_name in os.listdir(opt['input_folder']):
        yuv_path = os.path.join(opt['input_folder'], yuv_name)
        if not os.path.isfile(yuv_path):
            continue
        width, height = parse_resolution_from_filename(yuv_name)
        # 读取图像
        # image = cv2.imread(img_path)
        image = nv12_to_rgb(yuv_path, width, height)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # 对图像进行逐块预测
        sr_image = process_large_image_in_parts(image, opt['scale'], opt['patch_size'], opt['overlap'])

        # 保存结果
        sr_image = (sr_image * 255).astype(np.uint8)
        sr_image_bgr = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)



        # 归一化到 [0, 1]，然后还原到 [0, 255]
        rgb_image = image * 255.0  # 放大到 [0, 255]
        rgb_image = np.clip(rgb_image, 0, 255)  # Clip 避免溢出

        # 转为 BGR 格式
        image_lr = cv2.cvtColor(rgb_image.astype(np.uint8), cv2.COLOR_RGB2BGR)
        output_sr_path = os.path.join(opt['output_folder'], os.path.splitext(yuv_name)[0][:-1] + '_sr.png')
        output_lr_path = os.path.join(opt['output_folder'], os.path.splitext(yuv_name)[0][:-1] + '_lr.png')
        cv2.imwrite(output_sr_path, sr_image_bgr)
        cv2.imwrite(output_lr_path, image_lr)
        print(f"Saved super-resolution image to {output_sr_path}")
