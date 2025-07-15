import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


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

def visualize_latent_space(model, test_loader, method = "tsne", n_samples = 1000, device = "cuda", streamlit = False):
    model.eval()
    z_list = []
    y_list = []
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            z_mean, _ = model.encode(data)
            z_list.append(z_mean.cpu().numpy())
            y_list.append(labels.numpy())
    z_array = np.concatenate(z_list, axis = 0)[:n_samples]
    y_array = np.concatenate(y_list, axis = 0)[:n_samples]

    if method.lower() == "tsne":
        reducer = TSNE(n_components=3, random_state=42)
        z_2d = reducer.fit_transform(z_array)
        title = "TSNE Representation of Latent Space"

    else:
        reducer = PCA(n_components=3, random_state=42)
        z_2d = reducer.fit_transform(z_array)
        title = "PCA Representation of Latent Space"

    if streamlit:
        fig, ax = plt.subplots(figsize=(6, 6))
        scatter = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=y_array, cmap='viridis', s=5)
        plt.colorbar(scatter, ax=ax)
        ax.set_title(title)
        ax.set_xlabel(f"{method.upper()} Component 1")
        ax.set_ylabel(f"{method.upper()} Component 2")
        return fig
    else:
        plt.figure(figsize=(6, 6))
        plt.scatter(z_2d[:, 0], z_2d[:, 1], c=y_array, cmap='viridis', s=5)
        plt.colorbar()
        plt.title(title)
        plt.xlabel(f"{method.upper()} Component 1")
        plt.ylabel(f"{method.upper()} Component 2")
        plt.show()