import torch
import torch.nn as nn
import torch.utils as utils
import torch.nn.functional as F
import torch.optim as optim


class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes, img_channels = 3):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        self.label_embedding = nn.Embedding(self.num_classes, latent_dim)
        self.fc = nn.Linear(latent_dim * 2, 256 * 4 * 4)  # *2 because we concat noise + label
        self.fc_bn = nn.BatchNorm2d(256)

        self.model = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride = 2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, img_channels, kernel_size=3, padding = 1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        embedded_label = self.label_embedding(labels)
        x = torch.cat([noise, embedded_label], dim = 1)
        x = self.fc(x)
        x = x.view(-1, 256, 4, 4)
        x = self.fc_bn(x)
        x = self.model(x)

        return x


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
            nn.BatchNorm2d(128), 
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding = 1),
            nn.BatchNorm2d(128),  
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding = 1),
            nn.BatchNorm2d(256),  
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256 * 4 * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        embedded_label = self.label_embedding(labels)
        embedded_label = embedded_label.view(-1, 1, 32, 32)
        x = torch.cat([img, embedded_label], dim = 1)
        x = self.model(x)
        return x
    
class CGAN(nn.Module):
    def __init__(self, latent_dim = 100, num_classes = 10, img_channels = 3, img_size = 32, lr=0.0002, beta1=0.5, device = "cuda"):
        super(CGAN, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.img_channels = img_channels
        self.img_size = img_size
        self.device = device

        self.generator = Generator(latent_dim, num_classes, img_channels).to(device)
        self.discriminator = Discriminator(latent_dim, num_classes, img_channels, img_size).to(device)

        self.gen_optimizer = optim.Adam(self.generator.parameters(), lr = lr * 2, betas=(beta1, 0.999))  # 2x lr for generator
        self.dis_optimizer = optim.Adam(self.discriminator.parameters(), lr = lr, betas=(beta1, 0.999))

        self.criterion = nn.BCELoss()

        self.gen_losses = []
        self.dis_losses = []

    def to(self, device):
        """Move both generator and discriminator to device"""
        self.device = device
        self.generator = self.generator.to(device)
        self.discriminator = self.discriminator.to(device)
        return self
    
    def eval(self):
        """Set both networks to evaluation mode"""
        self.generator.eval()
        self.discriminator.eval()
        return self
    
    def train(self):
        """Set both networks to training mode"""  
        self.generator.train()
        self.discriminator.train()
        return self
    
    def load_state_dict(self, state_dict, strict = True):
        """Custom load_state_dict to handle CGAN checkpoint format"""
        if 'generator_state_dict' in state_dict:
            # Loading from CGAN checkpoint format
            self.generator.load_state_dict(state_dict['generator_state_dict'])
            self.discriminator.load_state_dict(state_dict['discriminator_state_dict'])
            if 'gen_optimizer_state_dict' in state_dict:
                self.gen_optimizer.load_state_dict(state_dict['gen_optimizer_state_dict'])
                self.dis_optimizer.load_state_dict(state_dict['dis_optimizer_state_dict'])
        else:
            # Use default behavior for regular state dict
            super().load_state_dict(state_dict, strict)
        
    def generate_noise(self, batch_size):
        return torch.randn(batch_size, self.latent_dim, device = self.device)
    
    def generate_labels(self, batch_size):
        return torch.randint(0, self.num_classes, (batch_size,), device= self.device )
    
    def train_discriminator(self, real_images, real_labels, batch_size):
        self.dis_optimizer.zero_grad()
        
        real_labels_tensor = torch.ones(batch_size, 1, device=self.device) * 0.9  # Label smoothing
        real_output = self.discriminator(real_images, real_labels)
        real_dis_loss = self.criterion(real_output, real_labels_tensor)
    
        #Creating fake images to train the discriminator        
        noise = self.generate_noise(batch_size)
        fake_labels = self.generate_labels(batch_size)
        fake_images = self.generator(noise, fake_labels)

        fake_labels_tensor = torch.ones(batch_size, 1, device=self.device) * 0.1  # Label smoothing
        fake_output = self.discriminator(fake_images.detach(), fake_labels)
        #Detach - breaks the flow between generator and discriminator 
        fake_dis_loss = self.criterion(fake_output, fake_labels_tensor)

        dis_loss = real_dis_loss + fake_dis_loss
        dis_loss.backward()
        self.dis_optimizer.step()

        return dis_loss.item(), real_dis_loss.item(), fake_dis_loss.item()
    
    def train_generator(self, batch_size):
        self.gen_optimizer.zero_grad()

        noise = self.generate_noise(batch_size)
        fake_labels = self.generate_labels(batch_size)
        fake_images = self.generator(noise, fake_labels)

        real_labels_tensor = torch.ones(batch_size, 1, device=self.device)
        output = self.discriminator(fake_images, fake_labels)
        gen_loss = self.criterion(output, real_labels_tensor)

        gen_loss.backward()
        self.gen_optimizer.step()

        return gen_loss.item()
    
    def generate_samples(self, num_samples, specific_class=None):
        """Generate samples for visualization"""
        self.generator.eval()
        with torch.no_grad():
            noise = self.generate_noise(num_samples)
            if specific_class is not None:
                labels = torch.full((num_samples,), specific_class, device=self.device)
            else:
                labels = self.generate_labels(num_samples)
            
            fake_images = self.generator(noise, labels)
            return fake_images, labels
