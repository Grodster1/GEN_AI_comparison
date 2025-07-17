import torch
import os
import sys
import torch.nn.functional as F
import torch.optim as optim
from data_prep import get_cifar10
from vae_model import VAE, vae_loss
from cvae_model import ConditionalVAE
from visualization import show_reconstructions, sample_latent_space
from utils import save_checkpoint, load_checkpoint

def train_vae(model:VAE, train_loader, test_loader, epochs = 50, lr=1e-4, beta = 1.0, device = 'cuda', checkpoint_path = None):
    print(f"Using {device}")

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr = lr)

    start_epoch = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        start_epoch = load_checkpoint(model, optimizer, filename = checkpoint_path, device = device)

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mean, logvar = model(data)
            loss = vae_loss(recon_batch, data, mean, logvar, beta)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs} [{batch_idx*len(data)}/{len(train_loader.dataset)}] "
                      f"Loss: {loss.item()/len(data):.6f}")
                
        print(f"Epoch {epoch+1}/{epochs} Average Loss: {train_loss/len(train_loader.dataset):.6f}")
        
        if (epoch + 1) % 10 == 0:
            show_reconstructions(model, test_loader, device)
            sample_latent_space(model, latent_dim=model.latent_dim, device=device)
            save_checkpoint(model, optimizer, epoch, f"checkpoints/vae_epoch_{epoch+1}.pth")

    save_checkpoint(model, optimizer, epochs, f"checkpoints/vae_final.pth")

def train_cvae(model:ConditionalVAE, train_loader, test_loader, epochs = 50,  lr=1e-4, beta = 1.0, device = 'cuda', checkpoint_path = None):
    print(f"Using {device}")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr = lr)

    start_epoch = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        start_epoch = load_checkpoint(model, optimizer, checkpoint_path, device)

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0
        for batch_idx, (data, label) in enumerate(train_loader):
            label_onehot = F.one_hot(label, num_classes=10).float().to(device)
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mean, logvar = model(data, label_onehot)
            loss = vae_loss(recon_batch, data, mean, logvar, beta)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs} [{batch_idx*len(data)}/{len(train_loader.dataset)}] "
                      f"Loss: {loss.item()/len(data):.6f}")
                
        print(f"Epoch {epoch+1}/{epochs} Average Loss: {train_loss/len(train_loader.dataset):.6f}")
        
        if (epoch + 1) % 10 == 0:
            #show_reconstructions(model, test_loader, device)
            #sample_latent_space(model, latent_dim=model.latent_dim, device=device)
            save_checkpoint(model, optimizer, epoch, f"checkpoints/cvae_epoch_{epoch+1}.pth")

    save_checkpoint(model, optimizer, epochs, f"checkpoints/cvae_final.pth")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "vae":
        torch.manual_seed(42)
        train_loader, test_loader, classes = get_cifar10(batch_size=64)
        model = VAE(latent_dim=128)
        train_vae(model, train_loader, test_loader, epochs=50, lr=5e-5, beta=1.0, 
                    device="cuda" if torch.cuda.is_available() else "cpu", 
                    checkpoint_path="checkpoints/vae_final.pth" if os.path.exists("checkpoints/vae_final.pth") else None)
        
    elif len(sys.argv) > 1 and sys.argv[1] == "cvae":
        torch.manual_seed(42)
        train_loader, test_loader, classes = get_cifar10(batch_size=64)
        model = ConditionalVAE(latent_dim=128)
        train_cvae(model, train_loader, test_loader, epochs=40, lr=1e-4, beta=1.0, 
                    device="cuda" if torch.cuda.is_available() else "cpu", 
                    checkpoint_path="checkpoints/cvae_final.pth" if os.path.exists("checkpoints/cvae_final.pth") else None)
    else:
        print("Enter 'vae' or 'cvae' in the command")