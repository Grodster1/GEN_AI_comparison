import torch 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

def get_cifar10(batch_size = 64, data_dir = "./data"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(root = data_dir, train = True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(root = data_dir, train = False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle = True, num_workers = 2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle = False, num_workers=2)

    return train_loader, test_loader, train_dataset.classes


if __name__ == "__main__":
    train_loader, test_loader, classes = get_cifar10()
    print(f"Classes: {classes}")
    print(f"Train Loader Size: {len(train_loader.dataset)}")
    print(f"Test Loader Size: {len(test_loader.dataset)}")