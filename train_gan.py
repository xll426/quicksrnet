import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import StepLR
from model import QuickSRNetSmall
from quickdata import SuperResolutionPairedDataset
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import matplotlib.pyplot as plt
import sys

# Ensure the current script directory is in the import path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from custom_loss import CharbonnierLoss, TVLoss, MSELoss, L1Loss,GANLoss

from discriminator_arch import UNetDiscriminatorSN
from srvgg_arch import SRVGGNetCompact

# Initialize each loss
charbonnier_loss = CharbonnierLoss(loss_weight=1.0)
tv_loss = TVLoss(loss_weight=1.0)
mse_loss = MSELoss(loss_weight=1.0)
l1_loss = L1Loss(loss_weight=1.0)
gan_loss = GANLoss(gan_type='vanilla')



def main():
    # Configuration options
    opt = {
        'dataroot_gt': 'D:\\SR\\DRealSR',
        'dataroot_lq': 'D:\\SR\\DRealSR',
        'gt_size': 640,
        'use_hflip': False,
        'use_rot': False,
        'scale': 8,
        'phase': 'train',
        'batch_size': 1,
        'epochs': 50,
        'lr': 1e-4,
        'save_dir': 'D:\\SR\\quicksr\\x8\\checkpoints_small_gan_',
        'resume': "D:\\SR\\quicksr\\x8\\checkpoints_small_gan\\best_quicksrnet.pth",
        'meta_info': ["D:\\SR\\DRealSR\\train_meta_info_8_deg_paired_.txt", "D:\\SR\\DRealSR\\val_meta_info_8_deg_paired_.txt"],
    }

    os.makedirs(opt['save_dir'], exist_ok=True)

    print("Loading datasets...")
    train_loader = DataLoader(
        SuperResolutionPairedDataset({
            'dataroot_gt': opt['dataroot_gt'],
            'dataroot_lq': opt['dataroot_lq'],
            'gt_size': opt['gt_size'],
            'scale': opt['scale'],
            'phase': 'train',
            'use_hflip': opt['use_hflip'],
            'use_rot': opt['use_rot'],
            'meta_info': opt['meta_info'][0]
        }),
        batch_size=opt['batch_size'], shuffle=True, num_workers=4, pin_memory=True
    )
    
    val_loader = DataLoader(
        SuperResolutionPairedDataset({
            'dataroot_gt': opt['dataroot_gt'],
            'dataroot_lq': opt['dataroot_lq'],
            'scale': opt['scale'],
            'phase': 'val',
            'use_hflip': False,
            'use_rot': False,
            'meta_info': opt['meta_info'][1]
        }),
        batch_size=1, shuffle=False, num_workers=4, pin_memory=True
    )
    print("Datasets loaded.")

    # Model, Discriminator, and Optimizer setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QuickSRNetSmall(scaling_factor=opt['scale']).to(device)
    discriminator = UNetDiscriminatorSN(num_in_ch=3, num_feat=64, skip_connection=True).to(device)
    optimizer_g = optim.Adam(model.parameters(), lr=opt['lr'])
    optimizer_d = optim.Adam(discriminator.parameters(), lr=opt['lr'])
    scheduler_g = StepLR(optimizer_g, step_size=10, gamma=0.5)
    scheduler_d = StepLR(optimizer_d, step_size=10, gamma=0.5)

    if opt['resume'] and os.path.exists(opt['resume']):
        checkpoint = torch.load(opt['resume'], map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer_g.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_psnr = checkpoint.get('best_psnr', 0)
        print(f"Resuming training from epoch {start_epoch}. Best PSNR so far: {best_psnr}")
    else:
        start_epoch = 0
        best_psnr = 0

    if torch.cuda.device_count() > 1:
        model = DDP(model, device_ids=[0])
        discriminator = DDP(discriminator, device_ids=[0])



    
    # 初始化图表数据
    all_batch_losses = []  # 用于保存所有 epoch 中的 batch loss
    epoch_loss_g_list = []      # 每个 epoch 的平均 epoch_loss_g
    epoch_loss_d_real_list = []      # 每个 epoch 的平均 epoch_loss_d_real
    epoch_loss_d_fake_list = []      # 每个 epoch 的平均 epoch_loss_d_fake

    val_psnrs = []         # 每个 epoch 的 PSNR
    val_ssims = []         # 每个 epoch 的 SSIM
    # 记录当前 batch 的编号偏移量
    batch_offset = 0
    def save_plots():
        if all_batch_losses:
            # 1. 所有 epoch 中的 batch loss 图
            plt.figure()
            batch_indices, batch_values = zip(*all_batch_losses)
            plt.plot(batch_indices, batch_values, 'b-', label='Batch Loss (all epochs)')
            plt.xlabel('Batch')
            plt.ylabel('Loss')
            plt.title('Batch Loss across All Epochs')
            plt.legend()
            plt.savefig(f"{opt['save_dir']}/all_epochs_batch_loss.png")
            plt.close()
        if epoch_loss_g:
            # 2. 每个 epoch 的平均 loss 图
            plt.figure()
            plt.plot(range(start_epoch, start_epoch + len(epoch_loss_g_list)), epoch_loss_g_list, 'g-', label='Epoch Loss_g')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Epoch Loss_g per Epoch')
            plt.legend()
            plt.savefig(f"{opt['save_dir']}/epoch_loss_g.png")
            plt.close()

        if epoch_loss_d_real:
            # 3. 每个 epoch 的平均 loss 图
            plt.figure()
            plt.plot(range(start_epoch, start_epoch + len(epoch_loss_d_real_list)), epoch_loss_d_real_list, 'g-', label='Epoch Loss_d_real')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Epoch Loss_d_real per Epoch')
            plt.legend()
            plt.savefig(f"{opt['save_dir']}/epoch_loss_d_real.png")
            plt.close()

        if epoch_loss_d_fake:
            # 4. 每个 epoch 的平均 loss 图
            plt.figure()
            plt.plot(range(start_epoch, start_epoch + len(epoch_loss_d_fake_list)), epoch_loss_d_fake_list, 'g-', label='Epoch Loss_d_fake')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Epoch Loss_d_fake per Epoch')
            plt.legend()
            plt.savefig(f"{opt['save_dir']}/epoch_loss_d_fake.png")
            plt.close()
        if val_psnrs:
            # 5. 每个 epoch 的 PSNR 图
            plt.figure()
            plt.plot(range(start_epoch, start_epoch + len(val_psnrs)), val_psnrs, 'r-', label='Validation PSNR')
            plt.xlabel('Epoch')
            plt.ylabel('PSNR')
            plt.title('Validation PSNR per Epoch')
            plt.legend()
            plt.savefig(f"{opt['save_dir']}/psnr_per_epoch.png")
            plt.close()
        if val_ssims:
            # 6. 每个 epoch 的 SSIM 图
            plt.figure()
            plt.plot(range(start_epoch, start_epoch + len(val_ssims)), val_ssims, 'm-', label='Validation SSIM')
            plt.xlabel('Epoch')
            plt.ylabel('SSIM')
            plt.title('Validation SSIM per Epoch')
            plt.legend()
            plt.savefig(f"{opt['save_dir']}/ssim_per_epoch.png")
            plt.close()

    def validate_model(model, val_loader, device):
        model.eval()
        total_psnr, total_ssim = 0.0, 0.0
        psnr_calculator = PeakSignalNoiseRatio(data_range=1.0).to(device)
        ssim_calculator = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

        with torch.no_grad():
            for batch in val_loader:
                lr_images = batch['lq'].to(device)
                hr_images = batch['gt'].to(device)
                sr_images = model(lr_images)

                common_height, common_width = min(sr_images.shape[2], hr_images.shape[2]), min(sr_images.shape[3], hr_images.shape[3])
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
        model.train()
        discriminator.train()
        epoch_loss_g = 0.0
        epoch_loss_d_real, epoch_loss_d_fake = 0.0, 0.0

        for i, batch in enumerate(train_loader):
            lr_images = batch['lq'].to(device)
            hr_images = batch['gt'].to(device)

            # ======= Train Generator =======
            for p in discriminator.parameters():
                p.requires_grad = False

            optimizer_g.zero_grad()
            sr_images = model(lr_images)

            # Compute losses
            # charbonnier_loss_value = charbonnier_loss(sr_images, hr_images)
            # tv_loss_value = tv_loss(sr_images)
            # mse_loss_value = mse_loss(sr_images, hr_images)
            l1_loss_value = l1_loss(sr_images, hr_images)

            # Total generator loss
            # loss_g = (100 * charbonnier_loss_value + 1000 * tv_loss_value + 500 * mse_loss_value + 50 * l1_loss_value)
            loss_g = l1_loss_value
            all_batch_losses.append((batch_offset + i, l1_loss_value.item()))
            loss_g.backward()
            optimizer_g.step()

            # ======= Train Discriminator =======
            for p in discriminator.parameters():
                p.requires_grad = True

            optimizer_d.zero_grad()
            real_pred = discriminator(hr_images)
            fake_pred = discriminator(sr_images.detach())

            loss_d_real = gan_loss(real_pred, True, is_disc=True)
            loss_d_fake = gan_loss(fake_pred, False, is_disc=True)
            
            # Total discriminator loss
            loss_d_real.backward()
            loss_d_fake.backward()
            optimizer_d.step()

            # Log losses
            epoch_loss_g += loss_g.item()
            epoch_loss_d_real += loss_d_real.item()
            epoch_loss_d_fake += loss_d_fake.item()

            print(f"Batch {i+1}/{len(train_loader)}, Generator Loss: {loss_g.item():.4f}, "
                  f"Discriminator Real Loss: {loss_d_real.item():.4f}, Discriminator Fake Loss: {loss_d_fake.item():.4f}")

        # Average losses for the epoch
        avg_loss_g = epoch_loss_g / len(train_loader)
        avg_loss_d_real = epoch_loss_d_real / len(train_loader)
        avg_loss_d_fake = epoch_loss_d_fake / len(train_loader)
        epoch_loss_g_list.append(avg_loss_g)
        epoch_loss_d_real_list.append(avg_loss_d_real) 
        epoch_loss_d_fake_list.append(avg_loss_d_fake) 
        scheduler_g.step()
        scheduler_d.step()
        
        # Validate and save model if PSNR improves
        avg_psnr, avg_ssim = validate_model(model, val_loader, device)
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer_g.state_dict(),
                'best_psnr': best_psnr
            }, f"{opt['save_dir']}/best_quicksrnet.pth")
        


         # Save checkpoint after each epoch for both model and discriminator
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer_g.state_dict()
        }, f"{opt['save_dir']}/model_epoch_{epoch+1}.pth")

        torch.save({
            'epoch': epoch,
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizer_d_state_dict': optimizer_d.state_dict()
        }, f"{opt['save_dir']}/discriminator_epoch_{epoch+1}.pth")

        print(f"Epoch [{epoch+1}/{opt['epochs']}], Avg Generator Loss: {avg_loss_g:.4f}, "
              f"Avg Discriminator Real Loss: {avg_loss_d_real:.4f}, Avg Discriminator Fake Loss: {avg_loss_d_fake:.4f}, "
              f"PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}")
        val_psnrs.append(avg_psnr)
        val_ssims.append(avg_ssim)
        save_plots()  # 保存四张图


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()