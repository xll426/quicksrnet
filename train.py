import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import L1Loss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import StepLR
from model import QuickSRNetSmall
from quickdata import SuperResolutionPairedDataset
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import matplotlib.pyplot as plt
import sys
import os

# 确保当前脚本目录在导入路径中
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from custom_loss  import CharbonnierLoss, TVLoss, MSELoss, L1Loss

# 初始化各个损失
charbonnier_loss = CharbonnierLoss(loss_weight=1.0)
tv_loss = TVLoss(loss_weight=1.0)
mse_loss = MSELoss(loss_weight=1.0)
l1_loss = L1Loss(loss_weight=1.0)







def main():
    # 配置选项
    opt = {
        'dataroot_gt': 'D:\\SR\\molchip',
        'dataroot_lq': 'D:\\SR\\molchip',
        'gt_size': 380,
        'use_hflip': True,
        'use_rot': True,
        'scale': 2,
        'phase': 'train',
        'batch_size': 256,
        'epochs': 100,
        'lr': 1e-4,
        'save_dir': 'D:\\SR\\quicksr\\x2\\checkpoints_small\\molchip_y',
        'resume': None,
        'meta_info': ["D:\\SR\\molchip\\train_meta_info_2_molchip_paired.txt", "D:\\SR\\molchip\\val_meta_info_2_molchip_paired.txt"],
    }

    os.makedirs(opt['save_dir'], exist_ok=True)

    print("Loading datasets...")
    train_dataset = SuperResolutionPairedDataset({
        'dataroot_gt': opt['dataroot_gt'],
        'dataroot_lq': opt['dataroot_lq'],
        'gt_size': opt['gt_size'],
        'scale': opt['scale'],
        'phase': 'train',
        'use_hflip': opt['use_hflip'],
        'use_rot': opt['use_rot'],
        'meta_info': opt['meta_info'][0]
    })
    val_dataset = SuperResolutionPairedDataset({
        'dataroot_gt': opt['dataroot_gt'],
        'dataroot_lq': opt['dataroot_lq'],
        'scale': opt['scale'],
        'phase': 'val',
        'use_hflip': False,
        'use_rot': False,
        'meta_info': opt['meta_info'][1]
    })
    print("Datasets loaded.")

    train_loader = DataLoader(train_dataset, batch_size=opt['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QuickSRNetSmall(scaling_factor=opt['scale']).to(device)
    print(f"Model initialized on device: {device}")

    optimizer = optim.Adam(model.parameters(), lr=opt['lr'])
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    charbonnier_weight = 100.0
    tv_weight = 10000.0
    mse_weight = 1000.0
    l1_weight = 100.0

    # criterion = L1Loss()

    if opt['resume']:
        checkpoint = torch.load(opt['resume'], map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_psnr = checkpoint.get('best_psnr', 0)
        print(f"Resuming training from epoch {start_epoch}. Best PSNR so far: {best_psnr}")
    else:
        start_epoch = 0
        best_psnr = 0

    if torch.cuda.device_count() > 1:
        model = DDP(model, device_ids=[0])
        print(f"Training using {torch.cuda.device_count()} GPUs.")

    # 初始化图表数据
    all_batch_losses = []  # 用于保存所有 epoch 中的 batch loss
    epoch_losses = []      # 每个 epoch 的平均 loss
    val_psnrs = []         # 每个 epoch 的 PSNR
    val_ssims = []         # 每个 epoch 的 SSIM
    # 记录当前 batch 的编号偏移量
    batch_offset = 0


    def save_plots():
        # 确保保存目录存在
        os.makedirs(opt['save_dir'], exist_ok=True)
        
        # 1. 所有 epoch 中的 batch loss 图
        if all_batch_losses:
            plt.figure()
            batch_indices, batch_values = zip(*all_batch_losses)
            plt.plot(batch_indices, batch_values, 'b-', label='Batch Loss (all epochs)')
            plt.xlabel('Batch')
            plt.ylabel('Loss')
            plt.title('Batch Loss across All Epochs')
            plt.legend(loc="upper right")
            plt.savefig(f"{opt['save_dir']}/all_epochs_batch_loss.png")
            plt.close()

        # 2. 每个 epoch 的平均 loss 图
        if epoch_losses:
            plt.figure()
            plt.plot(range(start_epoch, start_epoch + len(epoch_losses)), epoch_losses, 'g-', label='Epoch Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Epoch Loss per Epoch')
            plt.legend(loc="upper right")
            plt.savefig(f"{opt['save_dir']}/epoch_loss.png")
            plt.close()

        # 3. 每个 epoch 的 PSNR 图
        if val_psnrs:
            plt.figure()
            plt.plot(range(start_epoch, start_epoch + len(val_psnrs)), val_psnrs, 'r-', label='Validation PSNR')
            plt.xlabel('Epoch')
            plt.ylabel('PSNR')
            plt.title('Validation PSNR per Epoch')
            plt.legend(loc="lower right")
            plt.savefig(f"{opt['save_dir']}/psnr_per_epoch.png")
            plt.close()

        # 4. 每个 epoch 的 SSIM 图
        if val_ssims:
            plt.figure()
            plt.plot(range(start_epoch, start_epoch + len(val_ssims)), val_ssims, 'm-', label='Validation SSIM')
            plt.xlabel('Epoch')
            plt.ylabel('SSIM')
            plt.title('Validation SSIM per Epoch')
            plt.legend(loc="lower right")
            plt.savefig(f"{opt['save_dir']}/ssim_per_epoch.png")
            plt.close()

    def validate_model(model, val_loader, device):
        model.eval()
        total_psnr, total_ssim = 0.0, 0.0
        psnr_calculator = PeakSignalNoiseRatio(data_range=1.0).to(device)
        ssim_calculator = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                lr_images = batch['lq'].to(device)
                hr_images = batch['gt'].to(device)
                sr_images = model(lr_images)
               
                _, _, h_sr, w_sr = sr_images.shape
                _, _, h_hr, w_hr = hr_images.shape
                common_height = min(h_sr, h_hr)
                common_width = min(w_sr, w_hr)
                sr_common = sr_images[:, :, :common_height, :common_width]
                hr_common = hr_images[:, :, :common_height, :common_width]

                psnr_value = psnr_calculator(sr_common, hr_common).item()
                ssim_value = ssim_calculator(sr_common, hr_common).item()

                total_psnr += psnr_value
                total_ssim += ssim_value

        avg_psnr = total_psnr / len(val_loader)
        avg_ssim = total_ssim / len(val_loader)
        print(f"Validation PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}")
        return avg_psnr, avg_ssim

    for epoch in range(start_epoch, opt['epochs']):
        print(f"Starting epoch {epoch + 1}/{opt['epochs']}...")
        model.train()
        epoch_loss = 0.0

        for i, batch in enumerate(train_loader):
            lr_images = batch['lq'].to(device)
           
            hr_images = batch['gt'].to(device)

            optimizer.zero_grad()
            sr_images = model(lr_images)
        
            # loss = criterion(sr_images, hr_images)
            # 假设你的损失函数计算如下
            # charbonnier_loss_value = charbonnier_loss(sr_images, hr_images)
            # tv_loss_value = tv_loss(sr_images)
            mse_loss_value = mse_loss(sr_images, hr_images)
            # l1_loss_value = l1_loss(sr_images, hr_images)

            # 计算加权总损失
            # loss_value = (charbonnier_weight * charbonnier_loss_value +
            #         tv_weight * tv_loss_value +
            #         mse_weight * mse_loss_value +
            #         l1_weight * l1_loss_value)
            loss_value = mse_loss_value

            loss_value.backward()

            # 打印每个损失项
            # print(f"Batch {i+1}/{len(train_loader)}: Charbonnier Loss: {charbonnier_loss_value.item():.4f}, "
            #     f"TV Loss: {tv_loss_value.item():.4f}, "
            #     f"MSE Loss: {mse_loss_value.item():.4f}, "
            #     f"L1 Loss: {l1_loss_value.item():.4f}, "
            #     f"Total Loss: {loss_value.item():.4f}")
            

            print(f"Batch {i+1}/{len(train_loader)}: "
                f"L1 Loss: {mse_loss_value.item():.4f}, "
                f"Total Loss: {loss_value.item():.4f}")
            optimizer.step()

            all_batch_losses.append((batch_offset + i, loss_value.item()))
            epoch_loss += loss_value.item()
            print(f"Epoch [{epoch+1}/{opt['epochs']}], Batch [{i+1}/{len(train_loader)}], Loss: {loss_value.item():.4f}")
         # 更新 batch 偏移量
        batch_offset += len(train_loader)
        avg_loss = epoch_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{opt['epochs']}], Avg Loss: {avg_loss:.4f}")

        scheduler.step()

        avg_psnr, avg_ssim = validate_model(model, val_loader, device)
        val_psnrs.append(avg_psnr)
        val_ssims.append(avg_ssim)

        save_plots()  # 保存四张图

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_psnr': best_psnr,
            }
            torch.save(checkpoint, f"{opt['save_dir']}/best_quicksrnet.pth")
            print(f"New best model saved with PSNR: {best_psnr:.4f} at epoch {epoch+1}.")

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
