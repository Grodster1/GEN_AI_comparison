from vae_model import VAE, vae_loss
import torch
from torch import nn
import torch.nn.functional as F

class ConditionalVAE(VAE):
    def __init__(self, latent_dim = 128, num_classes = 10):
        super().__init__(latent_dim)
        self.num_classes = num_classes
        self.label_encoder = nn.Sequential(
            nn.Linear(num_classes, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        self.fc_mean = nn.Linear(512 + 256, latent_dim)
        self.fc_logvar = nn.Linear(512 + 256, latent_dim)

        self.decoder_input = nn.Linear(latent_dim + 256, 256 * 4 * 4)

    def encode(self, x, y):
        y_encoded = self.label_encoder(y)
        x_encoded = self.encoder(x)
        
        combined = torch.cat([x_encoded, y_encoded], dim = 1)

        mean = self.fc_mean(combined)
        logvar = self.fc_logvar(combined)

        return mean, logvar

    def condition_on_label(self, z, y):
        projected_label = self.label_encoder(y.float())
        return torch.cat([z, projected_label], dim = 1)
    
    def decode(self, z, y):
        y_encoded = self.label_encoder(y)
        z_combined = torch.cat([z, y_encoded], dim = 1)
        x = self.decoder_input(z_combined)
        x = x.view(-1, 256, 4, 4)
        return self.decoder(x)
    
    def forward(self, x, y):
        mean, logvar = self.encode(x, y)
        z = self.reparametrize(mean, logvar)
        return self.decode(z, y), mean, logvar

    def generate_samples(self, labels, device='cuda'):
        self.eval()
        with torch.no_grad():
            batch_size = len(labels)
            
            if isinstance(labels, list):
                labels = torch.tensor(labels, device=device)
            
            y_one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
            
            z = torch.randn(batch_size, self.latent_dim, device=device)
            
            samples = self.decode(z, y_one_hot)
            
            return samples