import torch 
import os

def save_checkpoint(model, optimizer, epoch, filename, is_cgan = False):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    if is_cgan:
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': model.generator.state_dict(),
            'discriminator_state_dict': model.discriminator.state_dict(),
            'gen_optimizer_state_dict': model.gen_optimizer.state_dict(),
            'dis_optimizer_state_dict': model.dis_optimizer.state_dict(),
            'latent_dim': model.latent_dim,
            'num_classes': model.num_classes
        }
    else:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
    torch.save(checkpoint, filename)
    print(f"Saved checkpoint: {filename}")


def load_checkpoint(model, optimizer, filename, device, is_cgan = False):
    checkpoint = torch.load(filename, map_location=device)
    if is_cgan:
        model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    
    epoch = checkpoint['epoch']
    print(f"Loaded checkpoint: {filename} (epoch {epoch})")
    return epoch