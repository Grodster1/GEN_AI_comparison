from vae_model import VAE, vae_loss
import torch
from torch import nn

class ConditionalVAE(VAE):
    def __init__(self, latent_dim = 128, num_classes = 10):
        super().__init__(latent_dim)
        self.num_classes = num_classes
        self.label_encoder = nn.Linear(num_classes, 512)

        self.fc_mean = nn.Linear(512 + 512, latent_dim)
        self.fc_logvar = nn.Linear(512 + 512, latent_dim)

        self.decoder_input = nn.Linear(latent_dim + 512, 256 * 4 * 4)

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
        z = torch.cat([z, y_encoded], dim = 1)
        x = self.decoder_input(z)
        x = x.view(-1, 256, 4, 4)
        return self.decoder(x)
    
    def forward(self, x, y):
        mean, logvar = self.encode(x, y)
        z = self.reparametrize(mean, logvar)
        return self.decode(z, y), mean, logvar
