from utils.idx2onehot import idx2onehot

import torch
import torchvision
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F

IMG_SIZE = 784

class VAE(nn.Module):
    def __init__(self, label_num):
        super(VAE, self).__init__()

        self.encoder = Encoder(label_num=label_num)
        self.decoder = Decoder(label_num=label_num)

    def reparameter(self, mu, var): # Sampling
        std = torch.exp(var/2)

        # randn_like : Return a Vector which mu = 0, var = 1, size = Size of std
        eps = torch.randn_like(std)

        return mu + std * eps

    def forward(self, input, label):
        mu, var = self.encoder(input, label)
        latent = self.reparameter(mu, var)
        output = self.decoder(latent, label)

        return output, mu, var

class Encoder(nn.Module):
    def __init__(self, label_num):
        super(Encoder, self).__init__()

        self.label_num = label_num

        # Onehot
        input_size = IMG_SIZE + label_num # 794

        # Encoder
        self.fc11 = nn.Linear(input_size, 128)
        self.fc12 = nn.Linear(128, 64)
        self.fc13_1 = nn.Linear(64, 32)
        self.fc13_2 = nn.Linear(64, 32)

        # Activate Function
        self.act_fn1 = nn.ReLU()
        self.act_fn2 = nn.Sigmoid()

    def forward(self, input, label):

        # Condition
        label = idx2onehot(label, self.label_num)
        x = torch.cat((input, label), dim=-1)

        # Forward
        x = self.fc11(x)
        x = self.act_fn1(x)
        x = self.fc12(x)
        x = self.act_fn1(x)

        mu = self.fc13_1(x)
        var = self.fc13_2(x)

        return mu, var

class Decoder(nn.Module):
    def __init__(self, label_num):
        super(Decoder, self).__init__()

        self.label_num = label_num

        # Onehot
        input_size = 32 + label_num

        # Decoder
        self.fc21 = nn.Linear(input_size, 64)
        self.fc22 = nn.Linear(64, 128)
        self.fc23 = nn.Linear(128, IMG_SIZE)

        # Activate Function
        self.act_fn1 = nn.ReLU()
        self.act_fn2 = nn.Sigmoid()

    def forward(self, z, label):

        # Condition
        label = idx2onehot(label, self.label_num)
        # [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]] & [[1, 0], [0, 1]]
        # -> [[0.1, 0.2, 0.3, 1, 0], [0.4, 0.5, 0.6, 0, 1]]
        z = torch.cat((z, label), dim=1)

        # Forward
        z = self.fc21(z)
        z = self.act_fn1(z)
        z = self.fc22(z)
        z = self.act_fn1(z)
        z = self.fc23(z)
        output = self.act_fn2(z)

        return output