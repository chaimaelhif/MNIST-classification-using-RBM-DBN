import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def pytorch_to_numpy(x):
    return x.detach().numpy()


def display_images(imgs):
    fig, axs = plt.subplots(1, imgs.shape[0])
    for j in range(imgs.shape[0]):
        axs[j].imshow(pytorch_to_numpy(imgs[j, 0, :,:]), cmap='gray')
        axs[j].axis('off')
    plt.show()


class VAE(torch.nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim, n_rows, n_cols, n_channels):
        super(VAE, self).__init__()

        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_channels = n_channels
        self.n_pixels = self.n_rows * self.n_cols
        self.z_dim = z_dim

        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1) 
        self.fc2 = nn.Linear(h_dim1, h_dim2) 
        self.fc31 = nn.Linear(h_dim2, z_dim) 
        self.fc32 = nn.Linear(h_dim2, z_dim) 
        # decoder part
        self.fc4 = nn.Linear(z_dim, h_dim2) 
        self.fc5 = nn.Linear(h_dim2, h_dim1) 
        self.fc6 = nn.Linear(h_dim1, x_dim) 

    def encoder(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h)) 
        return self.fc31(h), self.fc32(h)

    def decoder(self, z):
        h = F.relu(self.fc4(z)) 
        h = F.relu(self.fc5(h)) 
        return torch.sigmoid(self.fc6(h)).view(-1, self.n_channels, self.n_rows, self.n_cols)

    @staticmethod
    def sampling(mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        z_mu, z_log_var = self.encoder(x) 
        z = self.sampling(z_mu, z_log_var) 
        return self.decoder(z), z_mu, z_log_var

    @staticmethod
    def loss_function(x, y, mu, log_var):
        reconstruction_error = F.binary_cross_entropy(y, x)
        kld = 0.5*torch.sum(mu**2+log_var**2-1-torch.log(log_var**2))
        return reconstruction_error + kld

    def train_vae(self, data, batch_size=128, epochs=100, early_stopping=5):
        keep_track = 0
        loss = 100
        vae_optimizer = optim.Adam(self.parameters())
        for epoch in range(0, epochs):
            train_loss = 0
            data_copy = data.copy()
            np.random.shuffle(data_copy)
            for batch in range(0, data.shape[0], batch_size):
                data_batch = data_copy[
                             batch: min(batch + batch_size, data.shape[0]), :
                             ]
                data_batch = torch.tensor(data_batch, dtype=torch.float32)
                vae_optimizer.zero_grad()

                y, z_mu, z_log_var = self.forward(data_batch)
                loss_vae = self.loss_function(data_batch, nn.Flatten()(y), z_mu, z_log_var)
                loss_vae.backward()
                previous_loss = loss
                train_loss += loss_vae.item()
                loss = train_loss / data.shape[0]
                vae_optimizer.step()

            if keep_track < early_stopping and round(train_loss, 4) == round(previous_loss, 4):
                keep_track += 1
            elif keep_track == early_stopping:
                return self

            print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, loss))

    def generate_images_vae(self, n_images=5, plot=True):
        epsilon = torch.randn(n_images, 1, self.z_dim)
        imgs_generated = self.decoder(epsilon)
        if plot:
            display_images(imgs_generated)
        return imgs_generated
