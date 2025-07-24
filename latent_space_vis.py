import torch
from data_prep import get_cifar10
from vae_model import VAE
from cvae_model import ConditionalVAE
from cgan_model import CGAN
from visualization import visualize_latent_space, interpolate_latent_space, sample_latent_space
import os

def load_model(model, checkpoint_path, device):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            # Regular format (VAE, CVAE)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # CGAN format 
            model.load_state_dict(checkpoint)
            
        print(f"Loaded {checkpoint_path}")
    else:
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found!")
    
    model.eval()
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    _, test_loader, _ = get_cifar10(batch_size=64)
    cgan = CGAN().to(device)
    cgan = load_model(cgan, "checkpoints/cgan_epoch_50.pth", device)
    vae = VAE(latent_dim=128).to(device)
    vae = load_model(vae, "checkpoints/vae_final.pth", device)
    #cvae = ConditionalVAE(latent_dim=128).to(device)
    #cvae = load_model(cvae, "checkpoints/cvae_final.pth", device)
    #visualize_latent_space(vae, test_loader, method="tsne", n_samples=1000, device=device)
    #visualize_latent_space(vae, test_loader, method="pca", n_samples=1000, device=device)
    #interpolate_latent_space(vae, test_loader)
    sample_latent_space(model=cgan, latent_dim=cgan.latent_dim, label = 0, is_cgan=True)


if __name__ == "__main__":
    torch.manual_seed(42)
    main()