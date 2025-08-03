import torch
from data_prep import get_cifar10
from vae_model import VAE
from cvae_model import ConditionalVAE
from cgan_model import CGAN
from utils import save_checkpoint, load_checkpoint
from visualization import visualize_latent_space, interpolate_latent_space, sample_latent_space
import os

def load_model(model, checkpoint_path, device):
    """Load model checkpoint using existing utils functions"""
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        return 0  # Return epoch 0 if no checkpoint
    
    if isinstance(model, CGAN):
        # Handle CGAN case
        start_epoch = load_checkpoint(
            model,
            model.gen_optimizer,
            model.dis_optimizer,
            filename=checkpoint_path,
            device=device,
            is_cgan=True
        )
    else:
        # Handle VAE/CVAE case
        start_epoch = load_checkpoint(
            model,
            model.optimizer if hasattr(model, 'optimizer') else None,
            filename=checkpoint_path,
            device=device
        )
    
    model.to(device)
    model.eval()  # Set to evaluation mode
    print(f"Loaded model from {checkpoint_path} (epoch {start_epoch})")
    return start_epoch

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    """
    # Example for VAE
    print("\nGenerating VAE samples:")
    vae_model = VAE(latent_dim=128).to(device)
    load_model(vae_model, "checkpoints/vae_final.pth", device)
    sample_latent_space(vae_model, n_samples=10, device=device)
    
    # Example for CVAE
    print("\nGenerating CVAE samples (class 3):")
    cvae_model = ConditionalVAE(latent_dim=128).to(device)
    load_model(cvae_model, "checkpoints/cvae_final.pth", device)
    sample_latent_space(cvae_model, n_samples=10, device=device, label=3)
    """
    # Example for CGAN
    print("\nGenerating CGAN samples (class 5):")
    cgan_model = CGAN(num_classes=10, device=device).to(device)
    load_model(cgan_model, "cgan_checkpoints/cgan_epoch_50.pth", device)
    sample_latent_space(cgan_model, n_samples=10, device=device, label=5, is_cgan=True)

if __name__ == "__main__":
    main()