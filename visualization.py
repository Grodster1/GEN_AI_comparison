import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils


def imshow(img, title = ""):
    img = (img + 1) / 2
    npimg = img.numpy()
    plt.figure(figsize=(8,4))
    plt.imshow(np.transpose(npimg, (1,2,0))) #In torch we have (Channels, Height, Width) and Matplotlib expects (Height, Width, Channels)
    plt.title(title)
    plt.axis("off")
    plt.show()

def show_reconstructions(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        data, _ = next(iter(test_loader))
        data = data.to(device)
        recon, _, _ = model(data)
        comparison = torch.cat([data[:5], recon[:5]])
        imshow(vutils.make_grid(comparison.cpu(), nrow = 5), title="Original (Top) vs Reconstructed (Bottom)")

def sample_latent_space(model, latent_dim, n_samples=5, device = "cuda"):
    model.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim).to(device)
        samples = model.decode(z)
        imshow(vutils.make_grid(samples.cpu(), nrow=n_samples), title="Generated Samples")