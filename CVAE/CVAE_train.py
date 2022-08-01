# %%
import torch
import torchvision
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from CVAE_model import VAE

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMG_SIZE = 28*28
EPOCH = 10
BATCH = 16
LR = 1e-3

# %%
train_dataset = torchvision.datasets.MNIST(root='../datasets/mnist/',
                                           train=True,
                                           transform=torchvision.transforms.ToTensor(),
                                           )


train_loader = data.DataLoader(train_dataset,
                               batch_size=BATCH,
                               shuffle=True)

# %%
model = VAE(label_num=10).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

# %%
# Train the VAE
for epoch in range(EPOCH):
    for i, (train_datas, train_labels) in enumerate(tqdm(train_loader)):

        train_labels = train_labels.to(device)
        train_datas = train_datas.to(device)  # [BATCH, 1, 28, 28]
        train_datas = train_datas.view(-1, IMG_SIZE)  # [BATCH, 784]

        reconst, mu, var = model(train_datas, train_labels)

        # Reconsruct Loss & KL Loss
        reconstruct_loss = F.binary_cross_entropy(
            reconst, train_datas, size_average=False)
        kl_div = -0.5 * torch.sum(1 + var - mu.pow(2) - var.exp())

        # Backward
        total_loss = reconstruct_loss + kl_div
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    print("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}"
          .format(epoch+1, EPOCH, i+1, len(train_loader), reconstruct_loss.item(), kl_div.item()))

#%%
torch.save(model.state_dict(), 'save_model.pt')

# %%