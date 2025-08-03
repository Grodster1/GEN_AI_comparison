import torch
import torch.nn as nn
import torch.utils as utils
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class _Generator(nn.Module):
    def __init__(self, latent_dim = 100, num_classes = 10, img_channels = 3, img_size = 32, embedding_size = 30):
        super(_Generator, self).__init__()

        self.label_embedding = nn.Embedding(num_classes, embedding_size)
        self.init_size = img_size // 8
        self.fc = nn.Sequential(
                    nn.Linear(latent_dim + embedding_size, 512 * self.init_size * self.init_size),
                    nn.BatchNorm1d(512 * self.init_size * self.init_size),
                    nn.ReLU(True)
                )        
        self.model = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # 8x8 -> 16x16
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # Final conv to get correct channels
            nn.Conv2d(64, img_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

        self.apply(weights_init)

    def forward(self, noise, labels):
        embedded_labels = self.label_embedding(labels)

        x = torch.cat([noise, embedded_labels], dim = -1)
        x = self.fc(x)
        x = x.view(-1, 512, self.init_size, self.init_size)
        return self.model(x)
    
class _Discriminator(nn.Module):
    def __init__(self, num_classes = 10, img_channels =3, img_size = 32, embedding_size = 30):
        super(_Discriminator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, embedding_size)
        self.label_projector = nn.Sequential(
            nn.Linear(embedding_size, img_size * img_size),
            nn.LeakyReLU(0.2)
        )

        def discriminator_block(in_filters, out_filters):
            block = [
                nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_filters),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25)
            ]
            return block
        
        self.model = nn.Sequential(
            *discriminator_block(img_channels + 1, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512)
        )

        self.adv_layer = nn.Sequential(
                    nn.Linear(512 * 2 * 2, 1)
                )        
        self.apply(weights_init)

    def forward(self, img, labels):
        batch_size = labels.size(0)
        embedded_labels = self.label_embedding(labels)

        label_map = self.label_projector(embedded_labels)        
        label_map = label_map.view(batch_size, 1, img.size(2), img.size(3))
        
        
        x = torch.cat([img, label_map], dim=1)
        x = self.model(x)
        x = x.view(batch_size, -1)
        return self.adv_layer(x)
    

class CGAN(nn.Module):
    def __init__(self, latent_dim = 100, num_classes = 10, img_channels = 3, img_size = 32, embedding_size = 30, lr = 1e-4, device = 'cuda'):
        super(CGAN, self).__init__()
        self.device = device
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        self.generator = _Generator(latent_dim, num_classes, img_channels, img_size, embedding_size)
        self.discriminator = _Discriminator(num_classes, img_channels, img_size, embedding_size)

        self.generator.to(device)
        self.discriminator.to(device)

        self.gen_optimizer = optim.Adam(self.generator.parameters(), lr = lr, betas = (0.5, 0.999))
        self.dis_optimizer = optim.Adam(self.discriminator.parameters(), lr = lr, betas= (0.5, 0.999))

        self.gen_scheduler = optim.lr_scheduler.StepLR(self.gen_optimizer, step_size=20, gamma = 0.1)
        self.dis_scheduler = optim.lr_scheduler.StepLR(self.dis_optimizer, step_size=20, gamma=0.1)
        
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, noise, labels):
    
        return self.generator(noise, labels)

    def generate_noise(self, batch_size):
        return torch.randn(batch_size, self.latent_dim, device = self.device)
    
    def generate_labels(self, batch_size):
        return torch.randint(0, self.num_classes, (batch_size, ), device=self.device)
    
    def train_generator_step(self, batch_size):
        self.gen_optimizer.zero_grad()
        
        valid = torch.full((batch_size, 1), 1.0, device=self.device)
        noise = self.generate_noise(batch_size)

        gen_labels = self.generate_labels(batch_size)
        gen_imgs = self.generator(noise, gen_labels)

        fake_pred = self.discriminator(gen_imgs, gen_labels)
        g_loss = self.loss(fake_pred, valid)

        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)

        self.gen_optimizer.step()
        return g_loss
    

    def train_discriminator_step(self, real_imgs, real_labels):
        self.dis_optimizer.zero_grad()
        batch_size = real_imgs.size(0)
        valid = torch.full((batch_size, 1), 0.9, device = self.device, dtype=torch.float32)
        fake = torch.full((batch_size, 1), 0.0, device = self.device, dtype = torch.float32)

        real_pred = self.discriminator(real_imgs, real_labels)
        d_loss_real = self.loss(real_pred, valid)


        noise = self.generate_noise(batch_size)
        gen_imgs = self.generator(noise, real_labels)
        fake_pred = self.discriminator(gen_imgs.detach(), real_labels)
        d_loss_fake = self.loss(fake_pred, fake)

        d_loss = (d_loss_real + d_loss_fake) / 2
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)

        self.dis_optimizer.step()

        return d_loss, d_loss_real, d_loss_fake

   




