import torch
import os
import sys
import torch.nn.functional as F
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
            test_model_generation(model, device)
            save_checkpoint(model, optimizer, epoch, f"checkpoints/cvae_epoch_{epoch+1}.pth")

    save_checkpoint(model, optimizer, epochs, f"checkpoints/cvae_final.pth")


def train_cgan(model:CGAN, train_loader, test_loader, epochs = 200, checkpoint_path = None):
    cgan = model
    device = cgan.device
    print(f"Using {device}")

    
    start_epoch = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        start_epoch = load_checkpoint(cgan, None, checkpoint_path, device, is_cgan=True)

    # Learning rate schedulers
    g_scheduler = optim.lr_scheduler.StepLR(cgan.gen_optimizer, step_size=50, gamma=0.8)
    d_scheduler = optim.lr_scheduler.StepLR(cgan.dis_optimizer, step_size=50, gamma=0.8)

    for epoch in range(start_epoch, epochs):
        epoch_g_loss = 0
        epoch_d_loss = 0
        num_batches = 0
        
        for batch_idx, (real_images, real_labels) in enumerate(train_loader):
            real_images = real_images.to(device)
            real_labels = real_labels.to(device)
            batch_size = real_images.size(0)

            dis_loss, real_dis_loss, fake_dis_loss = cgan.train_discriminator(real_images, real_labels, batch_size)
            
            gen_loss1 = cgan.train_generator(batch_size)
            gen_loss2 = cgan.train_generator(batch_size)
            gen_loss = (gen_loss1 + gen_loss2) / 2
            
            epoch_g_loss += gen_loss
            epoch_d_loss += dis_loss
            num_batches += 1

            if batch_idx % 100 == 0:  # Print less frequently
                print(f"Epoch {epoch+1}/{epochs} [{batch_idx+1}/{len(train_loader)}] "
                      f"Gen Loss: {gen_loss:.4f}, Dis Loss: {dis_loss:.4f}, "
                      f"D(real): {real_dis_loss:.4f}, D(fake): {fake_dis_loss:.4f}")

        g_scheduler.step()
        d_scheduler.step()
        
        avg_g_loss = epoch_g_loss / num_batches
        avg_d_loss = epoch_d_loss / num_batches
        print(f"Epoch {epoch+1} Summary - Avg Gen Loss: {avg_g_loss:.4f}, Avg Dis Loss: {avg_d_loss:.4f}")
        
        if (epoch + 1) % 10 == 0:
            print("Generating sample images...")
            try:
                cgan.eval()
                with torch.no_grad():
                    for class_idx in range(10):
                        samples, _ = cgan.generate_samples(1, specific_class=class_idx)
                        print(f"Class {class_idx}: min={samples.min():.3f}, max={samples.max():.3f}, mean={samples.mean():.3f}")
                cgan.train()
            except Exception as e:
                print(f"Error generating samples: {e}")

        if (epoch + 1) % 25 == 0:
            save_checkpoint(cgan, None, epoch+1, f"checkpoints/cgan_epoch_{epoch+1}.pth", is_cgan=True)

    return cgan


def test_model_generation(model, device):
    """Test the model's generation capability"""
    model.eval()
    with torch.no_grad():
        # Generate one sample for each class
        for class_idx in range(10):
            samples = model.generate_samples([class_idx], device=device)
            print(f"Generated sample for class {class_idx}: "
                  f"min={samples.min():.3f}, max={samples.max():.3f}, "
                  f"mean={samples.mean():.3f}, std={samples.std():.3f}")
            
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
        train_cvae(model, train_loader, test_loader, epochs=60, lr=1e-4, beta=0.5, 
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    checkpoint_path="checkpoints/cvae_final.pth" if os.path.exists("checkpoints/cvae_final.pth") else None)
        
    elif len(sys.argv) > 1 and sys.argv[1] == "cgan":
        torch.manual_seed(42)
        train_loader, test_loader, classes = get_cifar10(batch_size=128)  
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = CGAN(device=device)
        train_cgan(model, train_loader, test_loader,
                    checkpoint_path="checkpoints/cgan_final.pth" if os.path.exists("checkpoints/cgan_final.pth") else None)
    else:
        print("Enter 'vae', 'cvae' or 'cgan' in the command")