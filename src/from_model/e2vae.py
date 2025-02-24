import torch
import torch.nn as nn
import torch.nn.functional as F
from e2cnn import gspaces, nn as e2nn
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from tqdm import tqdm



# Define E(2)-Equivariant Space
gspace = gspaces.Rot2dOnR2(N=12)  # 8 discrete rotations

class single_e2CNN_module(nn.Module):
    """
    e2CNN + BatchNorm + ReLU + 2stride_Pooling (optional)

    if headLayer:
        input: torch tensor
        output: GeometricTensor
    
    else:
        input: GeometricTensor
        output: GeoemtricTensor

    """
    def __init__(self, input_channels=1, output_channels=144, N_rotation=12, kernel_size=5, padding=2, sigma=0.66, 
                pooling: bool = False, headLayer: bool = True):
        super().__init__()

        self.N_rotation = N_rotation
        self.headLayer = headLayer
        self.r2_act = gspaces.Rot2dOnR2(N=N_rotation)

        if N_rotation != -1: # Discrete rotations
            assert output_channels % N_rotation == 0, "Output channels must be divisible by N_rotation"

        if headLayer:
            in_type = e2nn.FieldType(self.r2_act, input_channels*[self.r2_act.trivial_repr])
        else:
            assert input_channels % N_rotation == 0, "Input channels must be divisible by N_rotation if not headLayer" 
            in_type = e2nn.FieldType(self.r2_act, input_channels//N_rotation*[self.r2_act.regular_repr])

        out_type = e2nn.FieldType(self.r2_act, output_channels//N_rotation*[self.r2_act.regular_repr])     
        self.input_type = in_type   
        self.out_type = out_type


        self.cnn_block = e2nn.SequentialModule(
            e2nn.R2Conv(in_type, out_type, kernel_size=kernel_size, padding=padding, bias=False),
            e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type, inplace=True),
        )

        if pooling:
            self.pool = e2nn.PointwiseAvgPoolAntialiased(out_type, sigma=sigma, stride=2)

    def forward(self, x):
        if self.headLayer:
            assert isinstance(x, torch.Tensor), "Input must be a torch tensor for headLayer"
            x = e2nn.GeometricTensor(x, self.input_type)
        else:
            assert isinstance(x, e2nn.GeometricTensor), "Input tensor must be e2nn.GeometricTensor provided for non-headLayer"
        x = self.cnn_block(x)
        if hasattr(self, 'pool'):
            x = self.pool(x)
        return x


class EquivariantEncoder_double_cnn(nn.Module):
    def __init__(self, input_channels=1, N_rotation=12,
                hidden_cnn_channels: list=[96, 48, 48, 12], 
                hidden_pooling: list=[-1.00, 0.66, -1.00, 0.66],
                kernel_size=None, padding=None):
        super().__init__()

        self.input_channels = input_channels
        self.N_rotation = N_rotation
        self.hidden_cnn_channels = hidden_cnn_channels
        self.hidden_pooling = hidden_pooling
        self.hidden_pooling_bool = [False if i == -1 else True for i in hidden_pooling]
        self.kernel_size = kernel_size
        self.padding = padding
        self.spatical_compression_factor = np.sum(self.hidden_pooling_bool) * 2 

        assert len(hidden_cnn_channels) == len(hidden_pooling), "Length of hidden_cnn_channels and hidden_pooling must be equal"
        assert len(hidden_cnn_channels) > 0, "At least one hidden layer must be present"

        if kernel_size is None:
            self.kernel_size = [5] * len(hidden_cnn_channels)
        if padding is None:
            self.padding = [2] * len(hidden_cnn_channels)

        self.mu_cnn_stack = []
        self.logvar_cnn_stack = []
        for i in range(len(hidden_cnn_channels)):
            if i == 0:
                kwargs = dict(input_channels=input_channels, output_channels=hidden_cnn_channels[i],
                                        N_rotation=N_rotation, kernel_size=self.kernel_size[i], padding=self.padding[i],
                                        sigma=hidden_pooling[i], pooling=self.hidden_pooling_bool[i], headLayer=True)
                self.mu_cnn_stack.append(single_e2CNN_module(**kwargs))
                self.logvar_cnn_stack.append(single_e2CNN_module(**kwargs))
            else:
                kwargs = dict(input_channels=hidden_cnn_channels[i-1], output_channels=hidden_cnn_channels[i],
                                        N_rotation=N_rotation, kernel_size=self.kernel_size[i], padding=self.padding[i],
                                        sigma=hidden_pooling[i], pooling=self.hidden_pooling_bool[i], headLayer=False)
                self.mu_cnn_stack.append(single_e2CNN_module(**kwargs))
                self.logvar_cnn_stack.append(single_e2CNN_module(**kwargs))

        # self.mu_cnn_stack = e2nn.ModuleList(mu_cnn_stack)
        self.mu_cnn_stack = nn.ModuleList(self.mu_cnn_stack)
        self.logvar_cnn_stack = nn.ModuleList(self.logvar_cnn_stack)

        self.summary()
    def summary(self):
        print(f'Encoder hidden {len(self.hidden_cnn_channels)} layers:', [f"{self.input_channels}->"]+self.hidden_cnn_channels)
        print('Encoder pooling:', self.hidden_pooling)
        print('Encoder kernel_size:', self.kernel_size)
        print('Encoder padding:', self.padding)

        channel_compression_rate = self.input_channels / self.hidden_cnn_channels[-1]
        spatial_compression_rate = 1 / (np.sum(self.hidden_pooling_bool) * 4)
        print(f'Channel compression rate: {channel_compression_rate*100: .2f}%')
        print(f'Spatial compression rate: {spatial_compression_rate*100: .2f}%')
        print(f'Total compression rate: {channel_compression_rate * spatial_compression_rate*100: .2f}%')

        pass

    def forward(self, x):
        
        H, W = x.shape[-2:]
        assert H // self.spatical_compression_factor > 0 and W // self.spatical_compression_factor > 0, "Input size too small for the given hidden layers"

        mu = x
        logvar = x

        for i in range(len(self.mu_cnn_stack)):
            mu = self.mu_cnn_stack[i](mu)
            logvar = self.logvar_cnn_stack[i](logvar)
        return mu, logvar



##################### %%

# class e2CNN(nn.Module):
#     def __init__(self, input_channels=1, output_channels=128, N_rotation=12):
#         super().__init__()

#         self.N_rotation = N_rotation
#         self.r2_act = gspaces.Rot2dOnR2(N=N_rotation)
#         in_type = e2nn.FieldType(self.r2_act, input_channels*[self.r2_act.trivial_repr])
#         self.input_type = in_type

#         # convolution 1
#         out_type = e2nn.FieldType(self.r2_act, 24*[self.r2_act.regular_repr])
#         self.block1 = e2nn.SequentialModule(
#             # e2nn.MaskModule(in_type, 29, margin=1),
#             e2nn.R2Conv(in_type, out_type, kernel_size=7, padding=3, bias=False),
#             e2nn.InnerBatchNorm(out_type),
#             e2nn.ReLU(out_type, inplace=True)
#         )

#         # convolution 2
#         in_type = self.block1.out_type
#         out_type = e2nn.FieldType(self.r2_act, 48*[self.r2_act.regular_repr])
#         self.block2 = e2nn.SequentialModule(
#             e2nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
#             e2nn.InnerBatchNorm(out_type),
#             e2nn.ReLU(out_type, inplace=True)
#         )
#         self.pool1 = e2nn.SequentialModule(
#             e2nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
#         )        

#         # convolution 3
#         in_type = self.block2.out_type
#         out_type = e2nn.FieldType(self.r2_act, 48*[self.r2_act.regular_repr])
#         self.block3 = e2nn.SequentialModule(
#             e2nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
#             e2nn.InnerBatchNorm(out_type),
#             e2nn.ReLU(out_type, inplace=True)
#         )
        
#         # convolution 4
#         # the old output type is the input type to the next layer
#         in_type = self.block3.out_type
#         # the output type of the fourth convolution layer are 96 regular feature fields of C8
#         out_type = e2nn.FieldType(self.r2_act, 96*[self.r2_act.regular_repr])
#         self.block4 = e2nn.SequentialModule(
#             e2nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
#             e2nn.InnerBatchNorm(out_type),
#             e2nn.ReLU(out_type, inplace=True)
#         )
#         self.pool2 = e2nn.SequentialModule(
#            e2nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
#         )

#         # convolution 5
#         # the old output type is the input type to the next layer
#         in_type = self.block4.out_type
#         # the output type of the fifth convolution layer are 96 regular feature fields of C8
#         out_type = e2nn.FieldType(self.r2_act, (output_channels // 8)*[self.r2_act.regular_repr])
#         self.block5 = e2nn.SequentialModule(
#             e2nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
#             e2nn.InnerBatchNorm(out_type),
#             e2nn.ReLU(out_type, inplace=True)
#         )
        
#         self.pool3 = e2nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=1, padding=0)

#     def forward(self, input: torch.Tensor):
#         # wrap the input tensor in a GeometricTensor
#         # (associate it with the input type)
#         x = e2nn.GeometricTensor(input, self.input_type)
#         x = self.block1(x)
#         x = self.block2(x)
#         x = self.pool1(x)
#         x = self.block3(x)
#         x = self.block4(x)
#         x = self.pool2(x)
#         x = self.block5(x)
#         # x = self.block6(x)
#         # pool over the spatial dimensions
#         x = self.pool3(x)
#         # unwrap the output GeometricTensor
#         # (take the Pytorch tensor and discard the associated representation)
#         x = x.tensor

#         return x        

class c8CNNTranspose(nn.Module):
    def __init__(self, input_channels=128, output_channels=1):
        super().__init__()

        self.r2_act = gspace
        in_type = e2nn.FieldType(self.r2_act, (input_channels // 8) * [self.r2_act.regular_repr])
        self.input_type = in_type

        # Transposed convolution 1
        out_type = e2nn.FieldType(self.r2_act, 96 * [self.r2_act.regular_repr])
        self.block1 = e2nn.SequentialModule(
            e2nn.R2ConvTransposed(in_type, out_type, kernel_size=5, stride=2 ,padding=1, bias=False),
            e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type, inplace=True)
        )


        # Transposed convolution 2
        in_type = self.block1.out_type
        out_type = e2nn.FieldType(self.r2_act, 96 * [self.r2_act.regular_repr])
        self.block2 = e2nn.SequentialModule(
            e2nn.R2ConvTransposed(in_type, out_type, kernel_size=5,  stride=1, padding=2, bias=False),
            e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type, inplace=True)
        )


        # Transposed convolution 3
        in_type = self.block2.out_type
        out_type = e2nn.FieldType(self.r2_act, 48 * [self.r2_act.regular_repr])
        self.block3 = e2nn.SequentialModule(
            e2nn.R2ConvTransposed(in_type, out_type, kernel_size=5, stride=2, padding=0, bias=False),
            e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type, inplace=True)
        )

        # Transposed convolution 4
        in_type = self.block3.out_type
        out_type = e2nn.FieldType(self.r2_act, 48 * [self.r2_act.regular_repr])
        self.block4 = e2nn.SequentialModule(
            e2nn.R2ConvTransposed(in_type, out_type, kernel_size=5, stride=1, padding=2, bias=False),
            e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type, inplace=True)
        )


        # Transposed convolution 5
        in_type = self.block4.out_type
        out_type = e2nn.FieldType(self.r2_act, 24 * [self.r2_act.regular_repr])
        self.block5 = e2nn.SequentialModule(
            e2nn.R2ConvTransposed(in_type, out_type, kernel_size=5, stride=2, padding=2, bias=False),
            e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type, inplace=True)
        )

        # Final Transposed convolution
        in_type = self.block5.out_type
        out_type = e2nn.FieldType(self.r2_act, output_channels * [self.r2_act.trivial_repr])
        self.block6 = e2nn.SequentialModule(
            e2nn.R2ConvTransposed(in_type, out_type, kernel_size=7, stride=1, padding=2, bias=False),
            e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type, inplace=True)
        )


        # upsampling to 28x28
        self.upsample = nn.Upsample(size=28, mode='bilinear', align_corners=False)

    def forward(self, input: torch.Tensor):
        x = e2nn.GeometricTensor(input, self.input_type)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = x.tensor
        x = self.upsample(x)
        return x

# class EquivariantEncoder(nn.Module):
#     def __init__(self, input_channels=1, latent_dim=32):
#         super().__init__()
#         self.CNN_c8_mu = c8CNN(input_channels, output_channels=latent_dim)
#         # self.fc_mu = nn.Linear(128, latent_dim)
#         # self.fc_logvar = nn.Linear(latent_dim*3*3, latent_dim)
#         self.CNN_c8_sigma = c8CNN(input_channels, output_channels=latent_dim)

#     def forward(self, x):
#         mu = self.CNN_c8_mu(x)
#         logvar = self.CNN_c8_sigma(x)
#         # x = x.mean(dim=[2, 3])  # Global pooling
#         # mu = self.fc_mu(x)
#         # logvar = self.Relu1(self.fc_logvar(x))
#         return mu, logvar

class EquivariantDecoder(nn.Module):
    def __init__(self, output_channels=1, latent_dim=32):
        super().__init__()

        # self.fc = nn.Linear(latent_dim, latent_dim * 3 * 3)

        # Deconvolutions for upsampling
        self.latent_dim = latent_dim
        self.TCNN_c8 = c8CNNTranspose(input_channels=latent_dim, output_channels=output_channels)

    def forward(self, z):
        # Reshape to (batch_size, 32 * 8, 7, 7) to ensure proper upsampling
        # z = self.fc(z).view(-1, self.latent_dim, 3, 3)  
        x = self.TCNN_c8(z)

        return x


class VAE(nn.Module):
    def __init__(self, input_channels=1, latent_dim=32):
        super().__init__()
        self.encoder = EquivariantEncoder(input_channels, latent_dim)
        self.decoder = EquivariantDecoder(input_channels, latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std 

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

# Loss function
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction="sum")
    kl_loss = -0.02 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss


def print_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    print(f'Model size: {size_all_mb:.3f} MB')


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.randn(10, 1, 28, 28).to(device)

    c = single_e2CNN_module().to(device)
    vae = VAE().to(device)
    print_model_size(vae)
    enc = EquivariantEncoder_double_cnn().to(device)
    dec = EquivariantDecoder().to(device)