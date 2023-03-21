import matplotlib.pyplot as plt
import numpy as np

global n, p, x_im, y_im


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def logit(p):
    return np.log(p / (1 - p))


def lire_alpha_digit(df, index_digit):
    df = df[index_digit, :]
    df_bad = np.zeros((df.size, df[0, 0].size))
    k = 0  # image index
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            df_bad[k, :] = df[i, j].flatten()
            k += 1
    return df_bad


class RBM:
    def __init__(self, p, q, mu=0, sigma=0.1):
        self.a = np.zeros(p)
        self.b = np.zeros(q)
        self.W = np.random.normal(mu, sigma, size=(p, q))

    def entree_sortie_rbm(self, v):
        return sigmoid(self.b + sum(np.dot(v, self.W)))

    def sortie_entree_rbm(self, h):
        return sigmoid(self.a + sum(np.dot(h, self.W.T)))

    def train_rbm(self, x, epochs=100, learning_rate=0.1, batch_size=100):
        for i in range(epochs):
            x_copy = x.copy()
            np.random.shuffle(x_copy)

            for batch in range(0, x.shape[0], batch_size):
                x_batch = x_copy[batch: min(batch + batch_size, x.shape[0]), :]

                v0 = x_batch
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

            h_epoch = np.random.binomial(1, self.entree_sortie_rbm(x))
            x_rec = np.random.binomial(1, self.sortie_entree_rbm(h_epoch))
            mse = np.sum((x_rec - x) ** 2) / x.shape[0]
            print(f"Epoch({i}): mse={round(mse, 2)}")

        return self

    def generate_image_rbm(self, nb_data=1, nb_gibbs=100):
        images = np.zeros((nb_data, p))
        for data in range(nb_data):
            v = np.random.binomial(1, 0.5 * np.ones(p))
            for iter_gibbs in range(nb_gibbs):
                h = np.random.binomial(1, self.entree_sortie_rbm(v))
                v = np.random.binomial(1, self.sortie_entree_rbm(h))

            images[data, :] = v

            # Plot the generated images
            plt.figure()
            for i in range(len(images)):
                plt.plot(images[i, :].reshape((x_im, y_im)), cmap="gray")

        return images
