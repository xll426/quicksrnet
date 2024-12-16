import argparse
import glob
import os

# def parse_dimensions(filename):
#     """从文件名中提取长宽信息，不包含扩展名"""
#     base_name = os.path.basename(filename)
#     parts = base_name.split('_')
    
#     # 提取长和宽的数值（最后一部分是文件扩展名，忽略）
#     length = int(parts[1])  # LQ 或 GT 图像的长度部分
#     width = int(parts[2].split('.')[0])  # LQ 或 GT 图像的宽度部分，去掉扩展名
    
#     return length, width



def parse_dimensions(filename):
    """从文件名中提取长宽信息或图像类型（LR 或 HR），返回一个元组 (length, width) 或 None"""
    base_name = os.path.basename(filename)
    parts = base_name.split('_')
    # print(parts)

    # 判断是否是 x1 或 x2 的情况
    if 'x1' in parts:
        return 'LR'  # 低分辨率图像
    elif 'x2' in parts or 'x2.png' in parts:
        return 'HR'  # 高分辨率图像
    elif len(parts) == 1:  
        return 'HR_2'  # 只有一个部分时，默认为高分辨率图像（加上 x2）
    
    # 如果不是 x1/x2，尝试提取长宽信息
    try:
        length = int(parts[1])  # 长度部分
        width = int(parts[2].split('.')[0])  # 宽度部分（去掉扩展名）
        return length, width
    except (ValueError, IndexError):
        return None  # 如果解析失败，返回 None


def generate_meta_info(args):
    """生成图像对的 meta 信息文件。
    Args:
        args: 从命令行传入的参数，包括 GT 和 LQ 文件夹、meta 信息文件路径等。
    """
    # 获取 GT 和 LQ 图像路径
    img_paths_gt = glob.glob(os.path.join(args.gt_folder, '*'))
    img_paths_lq = glob.glob(os.path.join(args.lq_folder, '*'))

    # 使用文件名提取的长宽作为匹配的依据
    img_names_gt = {os.path.basename(path): path for path in img_paths_gt}
    img_names_lq = {os.path.basename(path): path for path in img_paths_lq}

    # 确保 GT 和 LQ 文件夹中的图像数量一致
    assert len(img_names_gt) == len(img_names_lq), (
        f'GT folder and LQ folder should have the same number of images, but got {len(img_names_gt)} and {len(img_names_lq)}.'
    )

    # 打开文件，用于写入 meta 信息
    with open(args.meta_info, 'w') as txt_file:
        # 逐个 GT 图像，确保它有对应的 LQ 图像
        for img_name_gt, img_path_gt in img_names_gt.items():
            # 获取 GT 图像的长宽
            dimensions = parse_dimensions(img_name_gt)

            if dimensions == 'HR':
                img_name_lq = img_name_gt.replace('x2', 'x1')
       

            elif dimensions == 'HR_2':
                img_name_lq = img_name_gt.split('.')[0]+"x2"+".png"


            elif dimensions:
                length_gt, width_gt = dimensions
                 # 查找对应的 LQ 图像，确保 LQ 图像的长宽是 GT 图像的一半
                expected_length_lq = length_gt // 2
                expected_width_lq = width_gt // 2

                # 查找对应的 LQ 图像
                img_name_lq = f"{img_name_gt.split('_')[0]}_{expected_length_lq}_{expected_width_lq}.png"
            else:
                img_name_lq = None  # 保持原文件名不变

           

            if img_name_lq in img_names_lq:
                img_path_lq = img_names_lq[img_name_lq]
                
                # 计算相对路径
                rel_path_gt = os.path.relpath(img_path_gt, args.gt_root)
                rel_path_lq = os.path.relpath(img_path_lq, args.lq_root)

                # 输出相对路径
                print(f'{rel_path_gt}, {rel_path_lq}')

                # 写入文件
                txt_file.write(f'{rel_path_gt}, {rel_path_lq}\n')
            else:
                print(f"Warning: No matching LQ image for {img_name_gt}.")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="生成图像对的 meta 信息文件")
    
    # 输入参数：GT 和 LQ 文件夹路径
    parser.add_argument(
        '--gt_folder', type=str, default=r'D:\\SR\\molchip\\val_2xHR_all',
        help='GT (高分辨率) 图像文件夹路径')
    
    parser.add_argument(
        '--lq_folder', type=str, default=r'D:\\SR\\molchip\\val_2xLR_all',
        help='LQ (低分辨率) 图像文件夹路径')
    
    # 根目录，用于生成相对路径
    parser.add_argument(
        '--gt_root', type=str, default=r"D:\\SR\\molchip",
        help='GT 文件夹的根目录，用于计算相对路径。如果为空，则使用 GT 文件夹的上一级目录。')
    
    parser.add_argument(
        '--lq_root', type=str, default=r"D:\\SR\\molchip",
        help='LQ 文件夹的根目录，用于计算相对路径。如果为空，则使用 LQ 文件夹的上一级目录。')
    
    # 输出的 meta 信息文件路径
    parser.add_argument(
        '--meta_info', type=str, default='D:\\SR\\molchip\\val_meta_info_2_molchip_paired.txt',
        help='meta 信息文件的输出路径')
    
    return parser.parse_args()

if __name__ == '__main__':
    # 解析参数
    args = parse_args()
    
    # 处理默认根目录
    if args.gt_root is None:
        args.gt_root = os.path.dirname(args.gt_folder)
    
    if args.lq_root is None:
        args.lq_root = os.path.dirname(args.lq_folder)

    # 创建保存 meta 信息的目录
    os.makedirs(os.path.dirname(args.meta_info), exist_ok=True)

    # 生成 meta 信息文件
    generate_meta_info(args)
