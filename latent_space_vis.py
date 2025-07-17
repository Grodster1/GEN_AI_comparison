import torch
from data_prep import get_cifar10
from vae_model import VAE
from visualization import visualize_latent_space, interpolate_latent_space
import os

def load_model(model, checkpoint_path, device):
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device)['model_state_dict'])
        print(f"Loaded {checkpoint_path}")
    else:
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found!")
    model.eval()
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    _, test_loader, _ = get_cifar10(batch_size=64)

    vae = VAE(latent_dim=128).to(device)
    vae = load_model(vae, "checkpoints/vae_final.pth", device)
    

    visualize_latent_space(vae, test_loader, method="tsne", n_samples=1000, device=device)
    visualize_latent_space(vae, test_loader, method="pca", n_samples=1000, device=device)
    interpolate_latent_space(vae, test_loader)


if __name__ == "__main__":
    torch.manual_seed(42)
    main()