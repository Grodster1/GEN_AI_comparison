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
        if getattr(m, 'bias', None) is not None:
            nn.init.constant_(m.bias.data, 0)

    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class _Generator(nn.Module):
    def __init__(self, latent_dim = 100, num_classes = 10, img_channels = 3, img_size = 32, embedding_size = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.embedding = nn.Embedding(num_classes, embedding_size)

        self.init_size = img_size // 8  # 4 for 32x32
        fc_dim = 512 * self.init_size * self.init_size
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + embedding_size, fc_dim),
            nn.BatchNorm1d(fc_dim),
            nn.ReLU(True)
        )

        self.deconv_blocks = nn.Sequential(
            # 4x4 -> 8x8
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

            # final conv
            nn.Conv2d(64, img_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

        self.apply(weights_init)

    def forward(self, noise, labels):
        lbl = self.embedding(labels)
        x = torch.cat([noise, lbl], dim=1)
        x = self.fc(x)
        x = x.view(x.size(0), 512, self.init_size, self.init_size)
        return self.deconv_blocks(x)
    
class _Discriminator(nn.Module):
    def __init__(self, num_classes = 10, img_channels =3, img_size = 32, embedding_size = 128):
        super().__init__()
        self.label_embedding = nn.Embedding(num_classes, embedding_size)

        def conv_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *conv_block(img_channels, 64, normalize=False),  # First layer shouldn't have BatchNorm
            *conv_block(64, 128),
            *conv_block(128, 256),
            *conv_block(256, 512)
        )

        self.feature_size = 512 * (img_size // 16) * (img_size // 16)  # 512 * 2 * 2 for 32x32
        
        self.fc_main = nn.Linear(self.feature_size, 1)
        
        self.embed_proj = nn.Linear(embedding_size, self.feature_size)

        self.apply(weights_init)

    def forward(self, img, labels):
        batch_size = labels.size(0)
        features = self.model(img)
        features = features.view(batch_size, -1)

        main_score = self.fc_main(features).view(-1)

        embedded = self.label_embedding(labels)
        embedded_proj = self.embed_proj(embedded)

        proj_score = torch.sum(features * embedded_proj, dim = 1)

        return main_score + proj_score

    

class CGAN(nn.Module):
    def __init__(self, latent_dim = 100, num_classes = 10, img_channels = 3, img_size = 32, embedding_size = 128, lr = 1e-4, device = 'cuda', betas=(0.0, 0.9), lambda_gp=10.0):
        super(CGAN, self).__init__()
        self.device = device
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.lambda_gp = lambda_gp


        self.generator = _Generator(latent_dim, num_classes, img_channels, img_size, embedding_size)
        self.discriminator = _Discriminator(num_classes, img_channels, img_size, embedding_size)

        #self.generator.to(device)
        #self.discriminator.to(device)

        self.gen_optimizer = optim.Adam(self.generator.parameters(), lr = lr, betas = betas)
        self.dis_optimizer = optim.Adam(self.discriminator.parameters(), lr = lr, betas = betas)

        #self.gen_scheduler = optim.lr_scheduler.StepLR(self.gen_optimizer, step_size=20, gamma = 0.1)
        #self.dis_scheduler = optim.lr_scheduler.StepLR(self.dis_optimizer, step_size=20, gamma=0.1)
        
        #self.loss = nn.BCEWithLogitsLoss()

    def forward(self, noise, labels):
        return self.generator(noise, labels)

    def generate_noise(self, batch_size):
        return torch.randn(batch_size, self.latent_dim, device = self.device)
    
    def generate_labels(self, batch_size):
        return torch.randint(0, self.num_classes, (batch_size, ), device=self.device)
    
    @staticmethod
    def _interpolate(real, fake):
        alpha = torch.rand(real.size(0), 1, 1, 1, device=real.device)
        alpha = alpha.expand_as(real)
        
        return (alpha * real + (1-alpha) * fake).requires_grad_(True)


    def gradient_penalty(self, real_imgs, fake_imgs, labels):
        interp = self._interpolate(real_imgs, fake_imgs)
        d_interp = self.discriminator(interp, labels)
        grads = torch.autograd.grad(
            outputs=d_interp,
            inputs=interp,
            grad_outputs=torch.ones_like(d_interp, device=self.device),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]  # [Batch, Channel, Height, Width]
        grads = grads.view(grads.size(0), -1)
        gp = ((grads.norm(2, dim=1) - 1) ** 2).mean()
        return gp
    

    def generator_loss(self, fake_imgs, fake_labels, instance_noise_std = 0.0):
        if instance_noise_std > 0.0:
            fake_imgs = fake_imgs + torch.randn_like(fake_imgs) * instance_noise_std
        fake_scores = self.discriminator(fake_imgs, fake_labels)
        return -fake_scores.mean()
    
    def discriminator_loss(self, real_imgs, real_labels, fake_imgs, fake_labels, instance_noise_std = 0.0):
        if instance_noise_std > 0.0:
            real_imgs = real_imgs + torch.randn_like(real_imgs) * instance_noise_std
            fake_imgs = fake_imgs + torch.randn_like(fake_imgs) * instance_noise_std
        
        real_scores = self.discriminator(real_imgs, real_labels)
        fake_scores = self.discriminator(fake_imgs, fake_labels)
        
        # WGAN discriminator loss: fake - real + lambda_gp * gp
        gp = self.gradient_penalty(real_imgs, fake_imgs, real_labels)
        d_loss = (fake_scores.mean() - real_scores.mean()) + self.lambda_gp * gp
        return d_loss, real_scores.mean().item(), fake_scores.mean().item(), gp.item()


    def train_generator_step(self, batch_size, fixed_labels = None, instance_noise_std = 0.0):
        self.gen_optimizer.zero_grad()
        
        noise = self.generate_noise(batch_size)
        if fixed_labels is not None:
            if len(fixed_labels) >= batch_size:
                labels = fixed_labels[:batch_size]
            else:
                labels = fixed_labels.repeat((batch_size // len(fixed_labels)) + 1)[:batch_size]
        else:
            labels = self.generate_labels(batch_size)

        gen_imgs = self.generator(noise, labels)
        g_loss = self.generator_loss(gen_imgs, labels, instance_noise_std)
        g_loss.backward()
        self.gen_optimizer.step()
        return g_loss.item()
    

    def train_discriminator_step(self, real_imgs, real_labels, instance_noise_std=0.0):
        self.dis_optimizer.zero_grad()
        batch_size = real_imgs.size(0)

        noise = self.generate_noise(batch_size)
        fake_labels = real_labels
        fake_imgs = self.generator(noise, fake_labels).detach()

        d_loss, r_mean, f_mean, gp = self.discriminator_loss(real_imgs, real_labels, fake_imgs, fake_labels, instance_noise_std)
        d_loss.backward()
        self.dis_optimizer.step()
        return d_loss.item(), r_mean, f_mean, gp

   




