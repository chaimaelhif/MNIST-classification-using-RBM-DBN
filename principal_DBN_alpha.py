import matplotlib.pyplot as plt
import numpy as np

from principal_RBM_alpha import RBM

global n, p, x_im, y_im


class DBN:
    def __init__(self, config=(p, 100, 50, 100)):
        self.config = config
        self.dbn = []
        for i in range(len(config) - 1):
            self.dbn.append(RBM(config[i], config[i + 1]))

    def train_dbn(self, x, epochs=100, learning_rate=0.1, batch_size=100):
        for i in range(len(self.config) - 1):
            self.dbn[i] = self.dbn[i].train_rbm(x, epochs, learning_rate, batch_size)
            x = np.random.binomial(1, self.dbn[i].entree_sortie_rbm(x))
        return self

    def generate_image_dbn(self, nb_data, nb_gibbs):
        v = self.dbn[-1].generate_image_rbm(nb_data, nb_gibbs)
        for i in range(len(self.config) - 1, -1):
            v = np.random.binomial(1, self.dbn[i].sortie_entree_rbm(v))

        images = v.copy()

        # Plot generated images
        plt.figure()
        for i in range(len(images)):
            plt.plot(images[i, :].reshape((x_im, y_im)), cmap="gray")

        return images
