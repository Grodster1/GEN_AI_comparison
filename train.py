import torch
import os
import sys
import numpy as np
import torch.nn.functional as F
from torchvision.utils import save_image
import torch.optim as optim
from data_prep import get_cifar10
from vae_model import VAE, vae_loss
from cvae_model import ConditionalVAE
from cgan_model import CGAN
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
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=3)
    
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs} [{batch_idx*len(data)}/{len(train_loader.dataset)}] "
                      f"Loss: {loss.item()/len(data):.6f}")

        avg_loss = train_loss / len(train_loader.dataset)       
        print(f"Epoch {epoch+1}/{epochs} Average Loss: {train_loss/len(train_loader.dataset):.6f}")
        scheduler.step(avg_loss)


        if (epoch + 1) % 10 == 0:
            #show_reconstructions(model, test_loader, device, n_samples = 5, cvae = True)
            #sample_latent_space(model, latent_dim=model.latent_dim, device=device)
            save_checkpoint(model, optimizer, epoch, f"checkpoints/cvae_epoch_{epoch+1}.pth")

    save_checkpoint(model, optimizer, epochs, f"checkpoints/cvae_final.pth")


def train_cgan(model: CGAN, train_loader, epochs=200, checkpoint_path=None):
    print(f"Using {model.device}")

    model.generator.to(model.device)
    model.discriminator.to(model.device)

    start_epoch = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        start_epoch = load_checkpoint(model, model.gen_optimizer, model.dis_optimizer, checkpoint_path, model.device, is_cgan=True)
        print(f"Resuming training from epoch {start_epoch + 1}")

    # Fixed samples for consistent visualization - 10 samples per class
    fixed_noise = model.generate_noise(100)
    fixed_labels = torch.cat([torch.full((10,), i) for i in range(model.num_classes)]).to(model.device)
    
    os.makedirs("cgan_samples", exist_ok=True)
    os.makedirs("cgan_checkpoints", exist_ok=True)

    d_losses = []
    g_losses = []

    for epoch in range(start_epoch, epochs):
        epoch_d_loss = 0
        epoch_g_loss = 0
        num_batches = len(train_loader)

        for batch_idx, (imgs, labels) in enumerate(train_loader):
            batch_size = imgs.size(0)
            imgs = imgs.to(model.device)
            labels = labels.to(model.device)

            # Train Discriminator
            d_loss, d_loss_real, d_loss_fake = model.train_discriminator_step(imgs, labels)
            
            # Adaptive Generator Training
            # Train generator 1-2 times based on discriminator performance
            if d_loss.item() < 0.6:  # Discriminator too strong
                g_steps = 2
            elif d_loss.item() > 1.2:  # Discriminator too weak  
                g_steps = 1
                # Skip one discriminator step occasionally
                if batch_idx % 3 == 0:
                    continue
            else:
                g_steps = 1
            
            g_loss_total = 0
            for _ in range(g_steps):
                g_loss = model.train_generator_step(batch_size)
                g_loss_total += g_loss.item()
            
            avg_g_loss = g_loss_total / g_steps
            
            epoch_d_loss += d_loss.item()
            epoch_g_loss += avg_g_loss

            if batch_idx % 100 == 0:
                print(
                    f"[Epoch {epoch+1}/{epochs}] [Batch {batch_idx}/{num_batches}] "
                    f"[D loss: {d_loss.item():.4f} (Real: {d_loss_real.item():.4f}, Fake: {d_loss_fake.item():.4f})] "
                    f"[G loss: {avg_g_loss:.4f}] [G steps: {g_steps}]"
                )

        # Calculate average losses
        avg_d_loss = epoch_d_loss / num_batches
        avg_g_loss = epoch_g_loss / num_batches
        d_losses.append(avg_d_loss)
        g_losses.append(avg_g_loss)

        print(f"Epoch {epoch+1} - Avg D Loss: {avg_d_loss:.4f}, Avg G Loss: {avg_g_loss:.4f}")

        # Update learning rates
        model.gen_scheduler.step()
        model.dis_scheduler.step()
        
        # Generate samples for visualization
        with torch.no_grad():
            model.generator.eval()
            samples = model(fixed_noise, fixed_labels).detach().cpu()
            save_image(samples, f"cgan_samples/{epoch+1:04d}.png", nrow=10, normalize=True)
            model.generator.train()
        
        # Save checkpoints
        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, model.gen_optimizer, model.dis_optimizer, epoch + 1, 
                          f"cgan_checkpoints/cgan_epoch_{epoch+1}.pth", is_cgan=True)

    # Save final checkpoint
    save_checkpoint(model, model.gen_optimizer, model.dis_optimizer, epochs, 
                   f"cgan_checkpoints/cgan_final.pth", is_cgan=True)
    
    return d_losses, g_losses


            
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
        trained_model, history = train_cvae(model, train_loader, test_loader, epochs=60, lr=1e-4, beta=0.5, 
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    checkpoint_path="checkpoints/cvae_final.pth" if os.path.exists("checkpoints/cvae_final.pth") else None)

    elif len(sys.argv) > 1 and sys.argv[1] == "cgan":
        torch.manual_seed(42)
        train_loader, classes = get_cifar10(batch_size=128, is_cgan=True) 
        print(f"Classes: {classes}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = CGAN(num_classes=len(classes), device=device)
        train_cgan(model, train_loader, epochs = 200, checkpoint_path="cgan_checkpoints/cgan_final.pth" if os.path.exists("cgan_checkpoints/cgan_final.pth") else None)
    else:
        print("Enter 'vae', 'cvae' or 'cgan' in the command")