import os
import cv2
import torch
import numpy as np
from model import QuickSRNetSmall,QuickSRNetMedium, QuickSRNetLarge  # 调用你定义的模型

# 配置选项
opt = {
    'scale': 8,  # 超分辨率缩放因子
    'model_checkpoint': 'D:\\SR\\quicksr\\x8\\checkpoints_small\\best_quicksrnet.pth',  # 模型检查点路径
    'input_folder': 'D:\\SR\\DRealSR\\test_LR\\8xLR',  # 输入图片文件夹路径
}



# 获取输入文件夹的路径，并在同一目录下创建一个输出文件夹
input_folder = opt['input_folder']
model_chepoint= opt['model_checkpoint']
output_folder = os.path.join(os.path.dirname(model_chepoint), f"{os.path.splitext(os.path.basename(opt['model_checkpoint']))[0]}_save")
os.makedirs(output_folder, exist_ok=True)


# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = QuickSRNetSmall(scaling_factor=opt['scale']).to(device)

# 加载预训练模型权重
checkpoint = torch.load(opt['model_checkpoint'], map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # 进入评估模式，不更新梯度

# 处理文件夹中的图片
def process_images(input_folder, output_folder):
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        if os.path.isfile(input_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            print(f"Processing {filename}...")

            # 读取并转换为 RGB 格式
            image = cv2.imread(input_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            # 转换为 PyTorch 张量
            image_tensor = torch.from_numpy(image_rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)

            # 模型推理
            with torch.no_grad():
                sr_image_tensor = model(image_tensor)

            # 转换回 NumPy 格式
            sr_image = sr_image_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            sr_image = (sr_image * 255).astype(np.uint8)

            # 保存推理后的图像
            output_path = os.path.join(output_folder, filename)
            sr_image_bgr = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, sr_image_bgr)

            print(f"Saved processed image to {output_path}")

# 主函数
if __name__ == '__main__':
    process_images(opt['input_folder'], output_folder)
