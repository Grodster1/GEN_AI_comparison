import torch
import torch.nn as nn
import torch.utils as utils
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes, img_channels = 3):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        self.label_embedding = nn.Embedding(self.num_classes, latent_dim)
        self.fc = nn.Linear(latent_dim * 2, 256 * 4 * 4)  # *2 because we concat noise + label

        self.model = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride = 2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, img_channels, kernel_size=3, padding = 1),
            nn.Tanh()
        )

class Discriminator(nn.Module):
    def __init__(self, latent_dim, num_classes, img_channels = 3, img_size = 32):
        super(Discriminator, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        self.label_embedding = nn.Embedding(self.num_classes, img_size*img_size)

        self.model = nn.Sequential(
            nn.Conv2d(img_channels + 1, 64, kernel_size=3, stride=1, padding = 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding = 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding = 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding = 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256 * 4 * 4, 1),
            nn.Sigmoid()
        )