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
gspace = gspaces.Rot2dOnR2(N=8)  # 8 discrete rotations

class c8CNN(nn.Module):
    def __init__(self, input_channels=1, output_channels=128):
        super().__init__()

        self.r2_act = gspace
        in_type = e2nn.FieldType(self.r2_act, input_channels*[self.r2_act.trivial_repr])
        self.input_type = in_type

        # convolution 1
        out_type = e2nn.FieldType(self.r2_act, 24*[self.r2_act.regular_repr])
        self.block1 = e2nn.SequentialModule(
            # e2nn.MaskModule(in_type, 29, margin=1),
            e2nn.R2Conv(in_type, out_type, kernel_size=7, padding=2, bias=False),
            e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type, inplace=True)
        )

        # convolution 2
        in_type = self.block1.out_type
        # the output type of the second convolution layer are 48 regular feature fields of C8
        out_type = e2nn.FieldType(self.r2_act, 48*[self.r2_act.regular_repr])
        self.block2 = e2nn.SequentialModule(
            e2nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type, inplace=True)
        )
        self.pool1 = e2nn.SequentialModule(
            e2nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )        

        # convolution 3
        # the old output type is the input type to the next layer
        in_type = self.block2.out_type
        # the output type of the third convolution layer are 48 regular feature fields of C8
        out_type = e2nn.FieldType(self.r2_act, 48*[self.r2_act.regular_repr])
        self.block3 = e2nn.SequentialModule(
            e2nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type, inplace=True)
        )
        
        # convolution 4
        # the old output type is the input type to the next layer
        in_type = self.block3.out_type
        # the output type of the fourth convolution layer are 96 regular feature fields of C8
        out_type = e2nn.FieldType(self.r2_act, 96*[self.r2_act.regular_repr])
        self.block4 = e2nn.SequentialModule(
            e2nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type, inplace=True)
        )
        self.pool2 = e2nn.SequentialModule(
           e2nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )

        # convolution 5
        # the old output type is the input type to the next layer
        in_type = self.block4.out_type
        # the output type of the fifth convolution layer are 96 regular feature fields of C8
        out_type = e2nn.FieldType(self.r2_act, (output_channels // 8)*[self.r2_act.regular_repr])
        self.block5 = e2nn.SequentialModule(
            e2nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type, inplace=True)
        )
        
        self.pool3 = e2nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=1, padding=0)

    def forward(self, input: torch.Tensor):
        # wrap the input tensor in a GeometricTensor
        # (associate it with the input type)
        x = e2nn.GeometricTensor(input, self.input_type)
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool1(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool2(x)
        x = self.block5(x)
        # x = self.block6(x)
        # pool over the spatial dimensions
        x = self.pool3(x)
        # unwrap the output GeometricTensor
        # (take the Pytorch tensor and discard the associated representation)
        x = x.tensor

        return x        

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

class EquivariantEncoder(nn.Module):
    def __init__(self, input_channels=1, latent_dim=32):
        super().__init__()
        self.CNN_c8_mu = c8CNN(input_channels, output_channels=latent_dim)
        # self.fc_mu = nn.Linear(128, latent_dim)
        # self.fc_logvar = nn.Linear(latent_dim*3*3, latent_dim)
        self.CNN_c8_sigma = c8CNN(input_channels, output_channels=latent_dim)

    def forward(self, x):
        mu = self.CNN_c8_mu(x)
        logvar = self.CNN_c8_sigma(x)
        # x = x.mean(dim=[2, 3])  # Global pooling
        # mu = self.fc_mu(x)
        # logvar = self.Relu1(self.fc_logvar(x))
        return mu, logvar

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


