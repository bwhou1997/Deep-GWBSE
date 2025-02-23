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
from .e2vae import VAE, EquivariantEncoder, EquivariantDecoder, vae_loss, print_model_size
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
# Define E(2)-Equivariant Space

# Data transformation: Normalize and convert to tensor
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure single channel
    transforms.ToTensor()
])

# Load MNIST dataset
train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=transform, download=True)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
# Print model size



import matplotlib.pyplot as plt

# Initialize model and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = VAE().to(device)


print_model_size(vae)
enc = EquivariantEncoder().to(device)
dec = EquivariantDecoder().to(device)
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)



if os.path.exists("vae_e2_mnist.pth"):
    vae.load_state_dict(torch.load("vae_e2_mnist.pth"), strict=False)
else:
    print("No pretrained model found.")
    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        vae.train()
        total_loss = 0

        for x, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            x = x.to(device)
            optimizer.zero_grad()
            x_recon, mu, logvar = vae(x)
            loss = vae_loss(x_recon, x, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader.dataset):.4f}")

    # Save model
    # torch.save(vae.state_dict(), "vae_e2_mnist.pth")

#######################
vae.eval()

# Get sample images from test set
with torch.no_grad():
    for x, _ in test_loader:
        x = x.to(device)
        x_recon, _, _ = vae(x)
        break  # Get only one batch

# Convert to NumPy
x = x.cpu().numpy()
x_recon = x_recon.cpu().numpy()

# Plot original vs reconstructed images
fig, axes = plt.subplots(2, 10, figsize=(10, 3))
for i in range(10):
    axes[0, i].imshow(x[i, 0], cmap="gray")
    axes[0, i].axis("off")
    axes[1, i].imshow(x_recon[i, 0], cmap="gray")
    axes[1, i].axis("off")

axes[0, 0].set_ylabel("Original")
axes[1, 0].set_ylabel("Reconstructed")
plt.show()



# %%


def test_rotation_equivariance(model, image, angle=45):
    model.eval()  # Set model to evaluation mode

    # Rotate the image
    rotated_image = TF.rotate(image, angle)

    # Pass both original and rotated image through the VAE
    with torch.no_grad():
        original_recon = model(image.unsqueeze(0))[0].squeeze(0)
        rotated_recon = model(rotated_image.unsqueeze(0))[0].squeeze(0)

    # Plot original, rotated input, and their reconstructions
    fig, axes = plt.subplots(2, 2, figsize=(6, 6))
    
    axes[0, 0].imshow(image.squeeze(0).cpu(), cmap="gray")
    axes[0, 0].set_title("Original Image")

    axes[0, 1].imshow(rotated_image.squeeze(0).cpu(), cmap="gray")
    axes[0, 1].set_title(f"Rotated Input ({angle}°)")

    axes[1, 0].imshow(original_recon.squeeze(0).cpu(), cmap="gray")
    axes[1, 0].set_title("Reconstruction")

    axes[1, 1].imshow(rotated_recon.squeeze(0).cpu(), cmap="gray")
    axes[1, 1].set_title(f"Reconstruction of Rotated Input")

    plt.tight_layout()
    plt.show()

# Example usage
sample_image = train_dataset[44][0].to(device)  # Get an MNIST sample (assuming dataset is loaded)
test_rotation_equivariance(vae, sample_image, angle=150)


# %%
