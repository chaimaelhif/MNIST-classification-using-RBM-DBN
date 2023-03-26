import matplotlib.pyplot as plt
import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def logit(z):
    return np.log(z / (1 - z))


class RBM:
    def __init__(self, num_visible, num_hidden=10, mu=0, sigma=0.1):
        """
        :param num_visible: number of neurons in visible layer
        :param num_hidden: number of neurons in the hidden layer
        :param mu: mean of the random variable to initialize Weights W
        :param sigma: std of the random variable to initialize Weights W
        """
        self.a: np.array = np.zeros(num_visible)
        self.b: np.array = np.zeros(num_hidden)
        self.W: np.array = np.random.normal(mu, sigma, size=(num_visible, num_hidden))

    def entree_sortie_rbm(self, v):
        return sigmoid(self.b + np.dot(v, self.W))

    def sortie_entree_rbm(self, h):
        return sigmoid(self.a + np.dot(h, self.W.T))

    def train_rbm(
        self, data, epochs=100, learning_rate=0.1, batch_size=100, early_stopping=5
    ):
        keep_track = 0
        mse = 100
        for epoch in range(epochs):
            data_copy = data.copy()
            np.random.shuffle(data_copy)

            for batch in range(0, data.shape[0], batch_size):
                data_batch = data_copy[
                    batch: min(batch + batch_size, data.shape[0]), :
                ]

                v0 = data_batch
                p_h_0 = self.entree_sortie_rbm(v0)
                h_0 = np.random.binomial(1, p_h_0)
                p_v_1 = self.sortie_entree_rbm(h_0)
                v1 = np.random.binomial(1, p_v_1)
                p_h_1 = self.entree_sortie_rbm(v1)

                grad_w = np.dot(v0.T, p_h_0) - np.dot(v1.T, p_h_1)
                grad_a = np.sum(v0 - v1, axis=0)
                grad_b = np.sum(p_h_0 - p_h_1, axis=0)

                self.W += learning_rate / batch_size * grad_w
                self.a += learning_rate / batch_size * grad_a
                self.b += learning_rate / batch_size * grad_b

            h_epoch = self.entree_sortie_rbm(data)
            data_rec = self.sortie_entree_rbm(h_epoch)
            previous_mse = mse
            mse = np.sum((data_rec - data) ** 2) / data.size

            if keep_track < early_stopping and round(mse, 3) == round(previous_mse, 3):
                keep_track += 1
            elif keep_track == early_stopping:
                return self
            print(f"Epoch {epoch+1}/{epochs} - Error: {mse:.3f}")

        return self

    def generate_image_rbm(self, x_im, y_im, nb_data=1, nb_gibbs=100, plot=False):
        p = self.W.shape[0]
        images = np.zeros((nb_data, p))
        for data in range(nb_data):
            v = np.random.binomial(1, 0.5 * np.ones(p))
            for iter_gibbs in range(nb_gibbs):
                h = np.random.binomial(1, self.entree_sortie_rbm(v))
                v = np.random.binomial(1, self.sortie_entree_rbm(h))

            images[data, :] = v

        if plot:
            # Reshape and Plot the generated images
            fig, axes = plt.subplots(1, nb_data, figsize=(10, 2))
            images = images.reshape((nb_data, x_im, y_im))
            for i in range(nb_data):
                axes[i].imshow(images[i], cmap="gray")
                axes[i].axis("off")
            plt.show()
        return images
