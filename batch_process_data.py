import os
import cv2

def crop_with_shift(image, block_size, base_name, output_dir):
    """
    裁剪图像，如果不足 block_size，起点向上或向左调整。
    
    Args:
        image (numpy.ndarray): 输入图像。
        block_size (int): 块的大小。
        base_name (str): 图像的基础名称。
        output_dir (str): 输出目录。

    Returns:
        None
    """
    h, w, c = image.shape
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            # 计算裁剪窗口的起点，向上或向左调整
            start_i = min(i, max(h - block_size, 0))
            start_j = min(j, max(w - block_size, 0))

            # 裁剪块
            crop = image[start_i:start_i + block_size, start_j:start_j + block_size]

            # 保存裁剪块
            save_name = f"{base_name}_{start_i}_{start_j}.png"
            cv2.imwrite(os.path.join(output_dir, save_name), crop)

def process_images(input_dir, gt_dir, output_input_dir, output_gt_dir, block_size=416):
    """
    处理输入和 GT 图像，确保文件对应并裁剪后保存。

    Args:
        input_dir (str): 输入图像目录。
        gt_dir (str): GT 图像目录。
        output_input_dir (str): 输出输入图像的裁剪目录。
        output_gt_dir (str): 输出 GT 图像的裁剪目录。
        block_size (int): 输入图像裁剪块大小。

    Returns:
        None
    """
    if not os.path.exists(output_input_dir):
        os.makedirs(output_input_dir)
    if not os.path.exists(output_gt_dir):
        os.makedirs(output_gt_dir)

    input_files = sorted(os.listdir(input_dir))
    gt_files = sorted(os.listdir(gt_dir))

    for input_file in input_files:
        if input_file[:-6]+".png" not in gt_files:
            print(f"跳过：{input_file} 没有对应的 GT 文件。")
            continue

        input_path = os.path.join(input_dir, input_file)
        gt_path = os.path.join(gt_dir, input_file[:-6]+".png")

        input_image = cv2.imread(input_path)
        gt_image = cv2.imread(gt_path)

        if input_image is None or gt_image is None:
            print(f"错误：无法读取文件 {input_file} 或其 GT 文件。")
            continue

        # 检查 GT 的尺寸是否是输入的两倍
        h_input, w_input, _ = input_image.shape
        h_gt, w_gt, _ = gt_image.shape
        assert h_gt == 2 * h_input and w_gt == 2 * w_input, \
            f"GT 文件的尺寸应为输入的两倍: {input_file}"

        # 裁剪输入和 GT 图像
        input_base_name = os.path.splitext(input_file)[0][:-2]
        crop_with_shift(input_image, block_size, input_base_name, output_input_dir)
        crop_with_shift(gt_image, block_size * 2, input_base_name, output_gt_dir)

if __name__ == "__main__":
    # 输入文件夹路径
    input_folder = "D:\\SR\\DIV2K_train_LR_unknown\\X2"  # 输入图像文件夹
    gt_folder = "D:\\SR\\DIV2K_train_HR"        # GT 图像文件夹

    # 输出文件夹路径
    output_input_folder = "D:\\SR\\DIV2K\\train_2xLR"  # 保存裁剪后的输入图像文件夹
    output_gt_folder = "D:\\SR\\DIV2K\\train_2xHR"        # 保存裁剪后的 GT 图像文件夹

    # 块大小
    block_size = 500  # 输入图像的块大小

    # 运行裁剪流程
    process_images(input_folder, gt_folder, output_input_folder, output_gt_folder, block_size)
