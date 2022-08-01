# %%
import torch
import torchvision
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from VAE_model import VAE

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMG_SIZE = 28*28
EPOCH = 15
BATCH = 16

# %%
# Test the VAE
test_dataset = torchvision.datasets.MNIST(root='datasets/mnist/',
                                          train=False,
                                          transform=torchvision.transforms.ToTensor(),
                                          )

test_loader = data.DataLoader(test_dataset,
                              batch_size=BATCH,
                              shuffle=True
                              )

# %%
model_path = 'model.pt'

model = VAE()
model.to(device)
model.load_state_dict(torch.load(model_path))

# %%
def show_images(images):
    plot_pos = int(np.sqrt(images.shape[0]))

    for index, image in enumerate(images):
        plt.subplot(plot_pos, plot_pos, index+1)
        plt.imshow(image.reshape(28, 28), cmap='gray')
        plt.axis('off')


# %%
with torch.no_grad():
    test_data = next(iter(test_loader))
    test_data = test_data[0].to(device).view(-1, IMG_SIZE)

    # Original Image
    show_images(test_data.cpu())
    plt.show()

    # Reconstrate Image
    outputs, _, _ = model(test_data)
    show_images(outputs.cpu())
    plt.show()

# %%
# Random Sample Vector to Generate
def generator():
    input = torch.randn(32)
    output = model.decoder(input.to(device))
    output = output.cpu().detach()

    plt.imshow(output.reshape(28, 28), cmap='gray')


generator()
# %%
