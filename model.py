import torch
import torch.nn as nn
from collections import OrderedDict

# Basic utility operations
class AddOp(nn.Module):
    def forward(self, x1, x2):
        return x1 + x2

class ConcatOp(nn.Module):
    def forward(self, *args):
        return torch.cat([*args], dim=1)

class AnchorOp(nn.Module):
    """
    Custom upsampling operation using convolutional layers
    to repeat interleaved values for the scaling factor.
    """
    def __init__(self, scaling_factor, in_channels=3, init_weights=True, freeze_weights=True, kernel_size=1, **kwargs):
        super().__init__()
        self.net = nn.Conv2d(in_channels=in_channels,
                             out_channels=in_channels * scaling_factor**2,
                             kernel_size=kernel_size,
                             **kwargs)
        if init_weights:
            num_channels_per_group = in_channels // self.net.groups
            # Initialize weights to perform nearest upsampling
            weight = torch.zeros(in_channels * scaling_factor**2, num_channels_per_group, kernel_size, kernel_size)
            bias = torch.zeros(weight.shape[0])
            for i in range(in_channels):
                weight[i * scaling_factor**2: (i + 1) * scaling_factor**2, i % num_channels_per_group, kernel_size // 2, kernel_size // 2] = 1.
            new_state_dict = OrderedDict({'weight': weight, 'bias':bias})
            self.net.load_state_dict(new_state_dict)

            if freeze_weights:
                for param in self.net.parameters():
                    param.requires_grad = False

    def forward(self, input):
        return self.net(input)

class PixelShufflev2(nn.Module):
    """
    A custom Pixel Shuffle operation for rearranging
    depth (channel) dimensions into spatial dimensions.
    """
    def __init__(self, scale_factor):
        super(PixelShufflev2, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        b, c, h, w = x.shape
        r = self.scale_factor
        x = x.view(b, c // (r * r), r, r, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        return x.view(b, c // (r * r), h * r, w * r)

# Base class for QuickSRNet
class QuickSRNetBase(nn.Module):
    """
    Base class for QuickSRNet variants.
    Supports integer scaling factors (2x, 3x, 4x, etc.)
    and specialized 1.5x scaling factor.
    """
    def __init__(self, scaling_factor, num_channels, num_intermediate_layers, use_ito_connection, in_channels=1, out_channels=1):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.out_channels = out_channels
        self.use_ito_connection = use_ito_connection

        intermediate_layers = []
        for _ in range(num_intermediate_layers):
            intermediate_layers.append(nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1))
            intermediate_layers.append(nn.Hardtanh(min_val=0., max_val=1.))

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=num_channels, kernel_size=3, padding=1),
            nn.Hardtanh(min_val=0., max_val=1.),
            *intermediate_layers
        )

        # Setup for the last conv layer
        self.conv_last = nn.Conv2d(num_channels, out_channels * (self.scaling_factor**2), kernel_size=3, padding=1)

        # Optional skip connection (Ito connection)
        if use_ito_connection:
            self.add_op = AddOp()
            self.anchor = AnchorOp(scaling_factor=self.scaling_factor,freeze_weights=False)

        # Custom pixel shuffle for scaling
        self.depth_to_space = PixelShufflev2(self.scaling_factor)
        self.clip_output = nn.Hardtanh(0., 1.)
        self.initialize()

    def forward(self, input):
        x = self.cnn(input)
        if self.use_ito_connection:
            residual = self.conv_last(x)
            input_upsampled = self.anchor(input)
            x = self.add_op(input_upsampled, residual)
        else:
            x = self.conv_last(x)
        x = self.clip_output(x)
        return self.depth_to_space(x)
    
    def initialize(self):
        for conv_layer in self.cnn:
            # Initialise each conv layer so that it behaves similarly to: 
            # y = conv(x) + x after initialization
            if isinstance(conv_layer, nn.Conv2d):
                middle = conv_layer.kernel_size[0] // 2
                num_residual_channels = min(conv_layer.in_channels, conv_layer.out_channels)
                with torch.no_grad():
                    for idx in range(num_residual_channels):
                        conv_layer.weight[idx, idx, middle, middle] += 1.

        if not self.use_ito_connection:
            # This will initialize the weights of the last conv so that it behaves like:
            # y = conv(x) + repeat_interleave(x, scaling_factor ** 2) after initialization
            middle = self.conv_last.kernel_size[0] // 2
            out_channels = self.conv_last.out_channels
            scaling_factor_squarred = out_channels // self.out_channels
            with torch.no_grad():
                for idx_out in range(out_channels):
                    idx_in = (idx_out % out_channels) // scaling_factor_squarred
                    self.conv_last.weight[idx_out, idx_in, middle, middle] += 1.

# QuickSRNet Small variant
class QuickSRNetSmall(QuickSRNetBase):
    def __init__(self, scaling_factor, **kwargs):
        super().__init__(scaling_factor, num_channels=32, num_intermediate_layers=2, use_ito_connection=False, **kwargs)

# QuickSRNet Medium variant
class QuickSRNetMedium(QuickSRNetBase):
    def __init__(self, scaling_factor, **kwargs):
        super().__init__(scaling_factor, num_channels=32, num_intermediate_layers=5, use_ito_connection=True, **kwargs)

# QuickSRNet Large variant
class QuickSRNetLarge(QuickSRNetBase):
    def __init__(self, scaling_factor, **kwargs):
        super().__init__(scaling_factor, num_channels=64, num_intermediate_layers=11, use_ito_connection=True, **kwargs)

# Model Factory to help initialize different models
def build_model(model_name, scaling_factor):
    if model_name == "quicksrnet_small":
        return QuickSRNetSmall(scaling_factor)
    elif model_name == "quicksrnet_medium":
        return QuickSRNetMedium(scaling_factor)
    elif model_name == "quicksrnet_large":
        return QuickSRNetLarge(scaling_factor)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
