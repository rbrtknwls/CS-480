from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from IPython.display import Image, display
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm

if not os.path.exists('results'):
    os.mkdir('results')

batch_size = 100
latent_size = 20

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(latent_size, 400),
            nn.ReLU(True),
            nn.Linear(400, 784),
            nn.Sigmoid()
        )

    def forward(self, z): \
            return self.main(z)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU(True),

            nn.Linear(400, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)


def train(generator, generator_optimizer, discriminator, discriminator_optimizer):
    # Trains both the generator and discriminator for one epoch on the training dataset.
    # Returns the average generator and discriminator loss (scalar values, use the binary cross-entropy appropriately)
    criterion = nn.BCELoss()

    real_label = 1.
    fake_label = 0.

    batch_gen_loss = []
    batch_dis_loss = []

    real_labels = torch.full((batch_size, 1), real_label)
    fake_labels = torch.full((batch_size, 1), fake_label)
    for images, _ in tqdm(train_loader):

        generator_optimizer.zero_grad()

        noise = torch.randn(batch_size, latent_size)

        generator.eval()
        fake_data = generator(noise)
        discriminator_output = discriminator(fake_data)

        generator_loss = criterion(discriminator_output, real_labels)

        generator_loss.backward()
        generator_optimizer.step()

        batch_gen_loss.append(generator_loss.item())

        discriminator_optimizer.zero_grad()

        real_images = images.reshape(-1, 784)

        generator.eval()
        with torch.no_grad():
            fake_data = generator(noise)

        true_predictions = discriminator(real_images)
        true_accuracy = criterion(true_predictions, real_labels)
        false_predictions = discriminator(fake_data)
        false_accuracy = criterion(false_predictions, fake_labels)

        discLoss = true_accuracy + false_accuracy
        discLoss.backward()
        discriminator_optimizer.step()

        batch_dis_loss.append(discLoss.item())

        last_image = real_images[0]
        gen_image = fake_data[0]

    return np.mean(batch_gen_loss), np.mean(batch_dis_loss)


def test(generator, discriminator):
    criterion = nn.BCELoss()

    real_label = 1.
    fake_label = 0.

    batch_gen_loss = []
    batch_dis_loss = []

    real_labels = torch.full((batch_size, 1), real_label)
    fake_labels = torch.full((batch_size, 1), fake_label)
    for images, _ in tqdm(test_loader):
        generator_optimizer.zero_grad()

        noise = torch.randn(batch_size, latent_size)

        generator.eval()
        fake_data = generator(noise)
        discriminator_output = discriminator(fake_data)

        generator_loss = criterion(discriminator_output, real_labels)

        generator_loss.backward()
        generator_optimizer.step()

        batch_gen_loss.append(generator_loss.item())

        discriminator_optimizer.zero_grad()

        real_images = images.reshape(-1, 784)

        generator.eval()
        with torch.no_grad():
            fake_data = generator(noise)

        true_predictions = discriminator(real_images)
        true_accuracy = criterion(true_predictions, real_labels)
        false_predictions = discriminator(fake_data)
        false_accuracy = criterion(false_predictions, fake_labels)

        discLoss = true_accuracy + false_accuracy
        discLoss.backward()
        discriminator_optimizer.step()

        batch_dis_loss.append(discLoss.item())

        last_image = real_images[0]
        gen_image = fake_data[0]

    return np.mean(batch_gen_loss), np.mean(batch_dis_loss)


epochs = 50

discriminator_avg_train_losses = []
discriminator_avg_test_losses = []
generator_avg_train_losses = []
generator_avg_test_losses = []

generator = Generator().to(device)
discriminator = Discriminator().to(device)

generator_optimizer = optim.Adam(generator.parameters(), lr=1e-3)
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=1e-3)

for epoch in range(1, epochs + 1):
    generator_avg_train_loss, discriminator_avg_train_loss = train(generator, generator_optimizer, discriminator,
                                                                   discriminator_optimizer)
    generator_avg_test_loss, discriminator_avg_test_loss = test(generator, discriminator)

    discriminator_avg_train_losses.append(discriminator_avg_train_loss)
    generator_avg_train_losses.append(generator_avg_train_loss)
    discriminator_avg_test_losses.append(discriminator_avg_test_loss)
    generator_avg_test_losses.append(generator_avg_test_loss)

    with torch.no_grad():
        sample = torch.randn(64, latent_size).to(device)
        sample = generator(sample).cpu()
        save_image(sample.view(64, 1, 28, 28),
                   'results/sample_' + str(epoch) + '.png')
        print('Epoch #' + str(epoch))
        display(Image('results/sample_' + str(epoch) + '.png'))
        print('\n')

plt.plot(discriminator_avg_train_losses)
plt.plot(generator_avg_train_losses)
plt.title('Training Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Disc', 'Gen'], loc='upper right')
plt.show()

plt.plot(discriminator_avg_test_losses)
plt.plot(generator_avg_test_losses)
plt.title('Test Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Disc', 'Gen'], loc='upper right')
plt.show()
