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
    """
    Training sequence for Variational Auto Encoder.
    
    Args:
        model (VAE model): The model to perform training on.
        train_loader: Data train loader with CIFAR10 dataset.
        test_loader: Data test loader with CIFAR10 dataset.
        epochs (int): Number of epochs of training.
        lr (float): Learning rate for training cycle.
        beta (float): Beta hyperparameter used for vae_loss function
        device (str): The device ('cuda' or 'cpu') to train model on.
        checkpoint_path (str): Path to saved checkpoint of trainig parameters.
    """
    
    
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


def train_cgan(model:CGAN, train_loader, epochs=200, n_critic=5, checkpoint_path=None, device='cuda'):
    
    print(f"Using {model.device}")
    model.generator.to(model.device)
    model.discriminator.to(model.device)

    start_epoch = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        start_epoch = load_checkpoint(model, model.gen_optimizer, model.dis_optimizer, checkpoint_path, model.device, is_cgan=True)
        print(f"Resuming trainig from epoch {start_epoch + 1}")

    fixed_noise = model.generate_noise(64)
    fixed_labels = torch.arange(model.num_classes).repeat(64 // model.num_classes + 1)[:64].to(model.device)
    
    os.makedirs("wcgan_samples", exist_ok=True)
    os.makedirs("wcgan_checkpoints", exist_ok=True)

    instance_noise_std = 0.1  
    noise_anneal = 0.995

    for epoch in range(start_epoch, epochs):

        epoch_d_loss = 0.0
        epoch_g_loss = 0.0

        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(model.device)
            labels = labels.to(model.device)
            d_loss_val = 0.0

            for _ in range(n_critic):
                d_loss, r_mean, f_mean, gp = model.train_discriminator_step(imgs, labels, instance_noise_std)
                d_loss_val = d_loss
            
            epoch_d_loss += d_loss_val

            g_loss_val = model.train_generator_step(imgs.size(0), fixed_labels[:imgs.size(0)], instance_noise_std)
            epoch_g_loss += g_loss_val

            if batch_idx % 100 == 0:
                print(f"[Epoch {epoch+1}/{epochs}] [Batch {batch_idx}/{len(train_loader)}] "
                      f"[D loss: {d_loss_val:.4f} (real_mean={r_mean:.4f}, fake_mean={f_mean:.4f}, gp={gp:.4f})] "
                      f"[G loss: {g_loss_val:.4f}]")

        instance_noise_std *= noise_anneal

        avg_d_loss = epoch_d_loss / len(train_loader)
        avg_g_loss = epoch_g_loss / len(train_loader)
        

        print(f"Epoch {epoch+1} - Avg D Loss: {avg_d_loss:.4f}, Avg G Loss: {avg_g_loss:.4f}")

        #model.gen_scheduler.step()
        #model.dis_scheduler.step()
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            with torch.no_grad():
                model.generator.eval()
                samples = model.generator(fixed_noise, fixed_labels).detach().cpu()
                save_image(samples, f"wcgan_samples/epoch_{epoch+1:04d}.png", nrow=8, normalize=True)
                model.generator.train()
        
        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, model.gen_optimizer, model.dis_optimizer, epoch + 1, f"wcgan_checkpoints/wcgan_epoch_{epoch+1}.pth", is_cgan=True)

    save_checkpoint(model, model.gen_optimizer, model.dis_optimizer, epochs, f"cgan_checkpoints/cgan_final.pth", is_cgan=True)


            
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
        model = CGAN(num_classes=len(classes), device=device, lr=1e-4, betas=(0.5,0.99), lambda_gp=5.0)        
        train_cgan(model, train_loader, epochs = 200, n_critic=4, checkpoint_path=None, device=device)
    else:
        print("Enter 'vae', 'cvae' or 'cgan' in the terminal")