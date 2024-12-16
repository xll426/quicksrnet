import os
import cv2

# 配置选项
opt = {
    'input_folder': 'D:\\SR\\DRealSR\\train_LR\\2xLR',  # 输入图像文件夹路径
    'output_folder': 'D:\\SR\\DRealSR\\train_LR',  # 输出文件夹根目录
    'scales': [2,4,8]  # 缩放因子
}

# 确保输出文件夹存在
for scale in opt['scales']:
    output_dir = os.path.join(opt['output_folder'], f'{scale+scale}xLR')
    os.makedirs(output_dir, exist_ok=True)

def downsample_image(img, scale):
    """简单的下采样函数，将图像缩小为指定倍数。"""
    h, w = img.shape[:2]
    img_lr = cv2.resize(img, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC)
    return img_lr

# 处理文件夹中的所有图像
def process_folder(input_folder, opt):
    for img_name in os.listdir(input_folder):
        img_path = os.path.join(input_folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图像: {img_name}")
            continue

        for scale in opt['scales']:
            img_lr = downsample_image(img, scale)
            # print(img_lr)
            output_path = os.path.join(opt['output_folder'], f'{scale+scale}xLR', img_name)
            cv2.imwrite(output_path, img_lr)
            print(f"已保存下采样图像: {output_path}")

# 主函数
if __name__ == "__main__":
    process_folder(opt['input_folder'], opt)
