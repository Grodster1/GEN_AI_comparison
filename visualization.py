import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch.nn.functional as F
import numpy as np
import torchvision.utils as vutils
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def imshow(img, title = "", streamlit = False):
    img = (img + 1) / 2
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1,2,0))
    if streamlit:
        return np.clip(npimg, 0, 1)
    else:
        fig = plt.figure(figsize=(8,4))
        plt.imshow(npimg) #In torch we have (Channels, Height, Width) and Matplotlib expects (Height, Width, Channels)
        plt.title(title)
        plt.axis("off")
        plt.show()

def show_reconstructions(model, test_loader, device, n_samples = 5, cvae=False):
    model.eval()
    with torch.no_grad():
        # Get a batch from the test loader
        data, label = next(iter(test_loader))
        data = data.to(device)
        
        if cvae:
            label_onehot = F.one_hot(label, num_classes=10).float().to(device)
            recon, _, _ = model(data, label_onehot)
        else:
            recon, _, _ = model(data)
        
        comparison = torch.cat([data[:n_samples], recon[:n_samples]])
        imshow(vutils.make_grid(comparison.cpu(), nrow=n_samples), title="Original (top) vs Reconstructed (bottom)")
    
def sample_latent_space(model, latent_dim, n_samples=1, device = "cuda", label = None):
    model.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim).to(device)
        if label is not None:
            y = torch.tensor([label] * n_samples, device=device)
            y_one_hot = F.one_hot(y, num_classes=model.num_classes).float()
            samples = model.decode(z, y_one_hot)
        else:
            samples = model.decode(z)
        imshow(vutils.make_grid(samples.cpu(), nrow=n_samples), title="Generated Samples")

def interpolate_latent_space(model, test_loader, n_steps=10, streamlit = False, device="cuda"):
    model.eval()
    with torch.no_grad():
        data, _ = next(iter(test_loader))
        data = data.to(device)
        z1, _ = model.encode(data[0:1])
        z2, _ = model.encode(data[1:2])
        
        alphas = torch.linspace(0, 1, n_steps, device=device).view(-1, 1)
        z_interp = alphas * z1 + (1 - alphas) * z2
        samples = model.decode(z_interp)
        return imshow(vutils.make_grid(samples.cpu(), nrow=n_steps), title="Latent Space Interpolation", streamlit = streamlit)


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
        z_3d = reducer.fit_transform(z_array)
        title = "TSNE Representation of Latent Space"

    else:
        reducer = PCA(n_components=3, random_state=42)
        z_3d = reducer.fit_transform(z_array)
        title = "PCA Representation of Latent Space"

    if streamlit:
        fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={"projection":"3d"})
        scatter = ax.scatter3D(z_3d[:, 0], z_3d[:, 1], z_3d[:, 2], c=y_array, cmap='viridis', s=5)
        ax.set_title(title)
        ax.set_xlabel(f"{method.upper()} Component 1")
        ax.set_ylabel(f"{method.upper()} Component 2")
        ax.set_zlabel(f"{method.upper()} Component 3")
        plt.colorbar(scatter, ax=ax, pad=0.1)


        return fig
    else:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter3D(z_3d[:, 0], z_3d[:, 1], z_3d[:, 2], c=y_array, cmap='viridis', s=5)
        ax.set_title(title)
        ax.set_xlabel(f"{method.upper()} Component 1")
        ax.set_ylabel(f"{method.upper()} Component 2")
        ax.set_zlabel(f"{method.upper()} Component 3")
        plt.colorbar(scatter, pad=0.1) 

        plt.show()