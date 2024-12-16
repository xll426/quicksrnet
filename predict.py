import cv2
import torch
import numpy as np
import ffmpeg
from skimage.metrics import peak_signal_noise_ratio as psnr
from model import QuickSRNetSmall  # 调用你定义的模型

# 配置选项
opt = {
    'scale': 2,  # 超分辨率缩放因子
    'model_checkpoint': 'D:\\SR\\quicksr\\x8\\checkpoints_small_L1loss\\checkpoint_float32.pth.tar',  # 模型检查点路径
    'output_video': './output_sr_ori_video.mp4',  # 输出超分辨率视频路径
    'input_video': './yuanquvideo_1.h265',  # 输入H265视频路径
    'resize_width': 640,  # 缩放宽度
    'resize_height': 360  # 缩放高度
}

# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = QuickSRNetSmall(scaling_factor=opt['scale']).to(device)

# 加载预训练模型权重
checkpoint = torch.load(opt['model_checkpoint'], map_location=device)
model.load_state_dict(checkpoint['state_dict'])
model.eval()  # 进入评估模式，不更新梯度

# 准备输出视频
def prepare_output_video(input_video_path, output_video_path, fps):
    # 使用 OpenCV 创建视频输出对象
    frame_width = opt['resize_width'] * opt['scale']
    frame_height = opt['resize_height'] * opt['scale']
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4 编码器
    out_stream = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    return out_stream

# 读取视频帧并推理
def process_video(input_video_path, output_video_path):
    input_stream = cv2.VideoCapture(input_video_path)
    fps = input_stream.get(cv2.CAP_PROP_FPS)  # 获取视频帧率
    out_stream = prepare_output_video(input_video_path, output_video_path, fps)

    frame_num = 0
    while input_stream.isOpened():
        ret, frame = input_stream.read()  # 读取一帧
        if not ret:
            break

        print(f"Processing frame {frame_num}...")

        # 缩放到指定大小
        frame_resized = cv2.resize(frame, (opt['resize_width'], opt['resize_height']))
        # frame_resized = frame
        print(frame_resized.shape)

        # 转换为 RGB 格式
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # 转换为 PyTorch 张量
        frame_tensor = torch.from_numpy(frame_rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)

        # 模型推理
        with torch.no_grad():
            sr_frame_tensor = model(frame_tensor)
        
        # 转换回 NumPy 格式
        sr_frame = sr_frame_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        sr_frame = (sr_frame * 255).astype(np.uint8)

        # 将推理后的帧写入输出视频
        sr_frame_bgr = cv2.cvtColor(sr_frame, cv2.COLOR_RGB2BGR)
        out_stream.write(sr_frame_bgr)

        frame_num += 1

    # 释放视频流
    input_stream.release()
    out_stream.release()
    print(f"Finished processing video. Output saved at {output_video_path}")

# 主函数
if __name__ == '__main__':
    process_video(opt['input_video'], opt['output_video'])
