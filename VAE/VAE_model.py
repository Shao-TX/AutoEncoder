import torch
import torchvision
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F

IMG_SIZE = 784

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # Encoder
        self.fc11 = nn.Linear(IMG_SIZE, 128)
        self.fc12 = nn.Linear(128, 64)
        self.fc13_1 = nn.Linear(64, 32)
        self.fc13_2 = nn.Linear(64, 32)

        # Decoder
        self.fc21 = nn.Linear(32, 64)
        self.fc22 = nn.Linear(64, 128)
        self.fc23 = nn.Linear(128, IMG_SIZE)

        # Activate Function
        self.act_fn1 = nn.ReLU()
        self.act_fn2 = nn.Sigmoid()

    def encoder(self, input):
        x = self.fc11(input)
        x = self.act_fn1(x)
        x = self.fc12(x)
        x = self.act_fn1(x)

        mu = self.fc13_1(x)
        var = self.fc13_2(x)

        return mu, var

    def decoder(self, z):
        z = self.fc21(z)
        z = self.act_fn1(z)
        z = self.fc22(z)
        z = self.act_fn1(z)
        z = self.fc23(z)
        output = self.act_fn2(z)

        return output

    def reparameter(self, mu, var): # Sampling
        std = torch.exp(var/2)

        # randn_like : Return a Vector which mu = 0, var = 1, size = Size of Input
        eps = torch.randn_like(std)

        return mu + std * eps

    def forward(self, input):
        mu, var = self.encoder(input)
        latent = self.reparameter(mu, var)
        output = self.decoder(latent)

        return output, mu, var