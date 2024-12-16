import torch
import torch.nn as nn
import torch.nn.functional as F


import math
import torch
from torch import autograd
from torch import nn
from torch.nn import functional as F

class GANLoss(nn.Module):
    """Define GAN loss.

    Args:
        gan_type (str): Support 'vanilla', 'lsgan', 'wgan', 'hinge'.
        real_label_val (float): The value for real label. Default: 1.0.
        fake_label_val (float): The value for fake label. Default: 0.0.
        loss_weight (float): Loss weight. Default: 1.0.
            Note that loss_weight is only for generators; and it is always 1.0
            for discriminators.
    """
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0, loss_weight=1.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type
        self.loss_weight = loss_weight
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan':
            self.loss = self._wgan_loss
        elif self.gan_type == 'wgan_softplus':
            self.loss = self._wgan_softplus_loss
        elif self.gan_type == 'hinge':
            self.loss = nn.ReLU()
        else:
            raise NotImplementedError(f'GAN type {self.gan_type} is not implemented.')

    def _wgan_loss(self, input, target):
        """wgan loss."""
        return -input.mean() if target else input.mean()

    def _wgan_softplus_loss(self, input, target):
        """wgan loss with soft plus."""
        return F.softplus(-input).mean() if target else F.softplus(input).mean()

    def get_target_label(self, input, target_is_real):
        """Get target label."""
        if self.gan_type in ['wgan', 'wgan_softplus']:
            return target_is_real
        target_val = (self.real_label_val if target_is_real else self.fake_label_val)
        return input.new_ones(input.size()) * target_val

    def forward(self, input, target_is_real, is_disc=False):
        """Forward function."""
        target_label = self.get_target_label(input, target_is_real)
        if self.gan_type == 'hinge':
            if is_disc:  # for discriminators in hinge-gan
                input = -input if target_is_real else input
                loss = self.loss(1 + input).mean()
            else:  # for generators in hinge-gan
                loss = -input.mean()
        else:  # other gan types
            loss = self.loss(input, target_label)

        # loss_weight is always 1.0 for discriminators
        return loss if is_disc else loss * self.loss_weight

class MultiScaleGANLoss(GANLoss):
    """MultiScaleGANLoss accepts a list of predictions."""
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0, loss_weight=1.0):
        super(MultiScaleGANLoss, self).__init__(gan_type, real_label_val, fake_label_val, loss_weight)

    def forward(self, input, target_is_real, is_disc=False):
        """Forward function."""
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]  # Only compute GAN loss for the last layer
                loss_tensor = super().forward(pred_i, target_is_real, is_disc).mean()
                loss += loss_tensor
            return loss / len(input)
        else:
            return super().forward(input, target_is_real, is_disc)

def r1_penalty(real_pred, real_img):
    """R1 regularization for discriminator."""
    grad_real = autograd.grad(outputs=real_pred.sum(), inputs=real_img, create_graph=True)[0]
    grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty

def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    """Path length regularization."""
    noise = torch.randn_like(fake_img) / math.sqrt(fake_img.shape[2] * fake_img.shape[3])
    grad = autograd.grad(outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True)[0]
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)
    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_lengths.detach().mean(), path_mean.detach()

def gradient_penalty_loss(discriminator, real_data, fake_data, weight=None):
    """Calculate gradient penalty for wgan-gp."""
    batch_size = real_data.size(0)
    alpha = real_data.new_tensor(torch.rand(batch_size, 1, 1, 1))

    # interpolate between real_data and fake_data
    interpolates = alpha * real_data + (1. - alpha) * fake_data
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = discriminator(interpolates)
    gradients = autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]

    if weight is not None:
        gradients = gradients * weight

    gradients_penalty = ((gradients.norm(2, dim=1) - 1)**2).mean()
    if weight is not None:
        gradients_penalty /= torch.mean(weight)

    return gradients_penalty






# Charbonnier Loss
class CharbonnierLoss(nn.Module):
    """Charbonnier loss (a differentiable variant of L1 Loss)."""
    def __init__(self, loss_weight=1.0, eps=1e-12, reduction='mean'):
        super(CharbonnierLoss, self).__init__()
        self.loss_weight = loss_weight
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred, target):
        diff = (pred - target)**2 + self.eps
        loss = torch.sqrt(diff)
        if self.reduction == 'mean':
            return self.loss_weight * loss.mean()
        elif self.reduction == 'sum':
            return self.loss_weight * loss.sum()
        else:
            return self.loss_weight * loss


# # Total Variation (TV) Loss
# class TVLoss(nn.Module):
#     """Total Variation Loss (used to encourage smoothness in image space)."""
#     def __init__(self, loss_weight=1.0):
#         super(TVLoss, self).__init__()
#         self.loss_weight = loss_weight

#     def forward(self, x):
#         batch_size = x.size(0)
#         h_tv = torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2).sum()
#         w_tv = torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2).sum()
#         return self.loss_weight * 2 * (h_tv + w_tv) / batch_size


class TVLoss(nn.Module):
    """Total Variation Loss (used to encourage smoothness in image space)."""
    def __init__(self, loss_weight=1.0):
        super(TVLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, x):
        batch_size = x.size(0)
        h_tv = torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2).mean()
        w_tv = torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2).mean()
        return self.loss_weight * (h_tv + w_tv)



# MSE Loss (L2 Loss)
class MSELoss(nn.Module):
    """Mean Squared Error (MSE) Loss."""
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target):
        loss = F.mse_loss(pred, target, reduction=self.reduction)
        return self.loss_weight * loss


# L1 Loss (Mean Absolute Error)
class L1Loss(nn.Module):
    """Mean Absolute Error (L1) Loss."""
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target):
        loss = F.l1_loss(pred, target, reduction=self.reduction)
        return self.loss_weight * loss
