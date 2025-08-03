import torch 
import os

def save_checkpoint(model, optimizer, second_optimizer=None, epoch=0, filename=None, is_cgan=False):
    """
    Saves a checkpoint at given epoch.

    Args:
        model (nn.Module): The model which parameters will be saved.
        optimizer (torch.optim.Optimizer): The primary optimizer to save.
        second_optimizer (torch.optim.Optimizer, optional): The secondary optimizer to save.
        epoch: Epoch at which parameters will be saved.
        filename (str): The path to the checkpoint file.
        device (str): The device ('cuda' or 'cpu') to map the checkpoint to.
        is_cgan (bool): A flag to indicate a CGAN model, which requires
                       separate loading for generator and discriminator.
    """
    if filename is None:
        print("Error: filename must be provided to save a checkpoint.")
        return
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    if is_cgan:
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': model.generator.state_dict(),
            'discriminator_state_dict': model.discriminator.state_dict(),
            'gen_optimizer_state_dict': optimizer.state_dict(),
            'dis_optimizer_state_dict': second_optimizer.state_dict(),
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



    


def load_checkpoint(model, optimizer, second_optimizer=None, filename=None, device='cpu', is_cgan=False):
    """
    Loads a checkpoint and resumes training from the last saved epoch.

    Args:
        model (nn.Module): The model to load the state into.
        optimizer (torch.optim.Optimizer): The primary optimizer to load.
        second_optimizer (torch.optim.Optimizer, optional): The secondary optimizer to load.
        filename (str): The path to the checkpoint file.
        device (str): The device ('cuda' or 'cpu') to map the checkpoint to.
        is_cgan (bool): A flag to indicate a CGAN model, which requires
                       separate loading for generator and discriminator.

    Returns:
        int: The epoch number from the loaded checkpoint.
    """

    if filename is None:
        print("Error: filename must be provided to load a checkpoint.")
        return 0
    checkpoint = torch.load(filename, map_location=device)
    if is_cgan:
        model.generator.load_state_dict(checkpoint['generator_state_dict'])
        model.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        if optimizer and second_optimizer:
            optimizer.load_state_dict(checkpoint['gen_optimizer_state_dict'])
            second_optimizer.load_state_dict(checkpoint['dis_optimizer_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint['epoch']
    print(f"Loaded checkpoint: {filename} (epoch {epoch})")
    return epoch