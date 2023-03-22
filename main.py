import numpy as np
import scipy

global n, p, x_im, y_im


def lire_alpha_digit(filename: str, indices: list):
    mat = scipy.io.loadmat(filename, simplify_cells=True)

    bad = mat['dat'][indices, :]

    x, y = bad[0, 0].shape
    images = np.zeros((bad.size, bad[0, 0].size))
    k = 0  # image index
    for i in range(bad.shape[0]):
        for j in range(bad.shape[1]):
            images[k, :] = bad[i, j].flatten()
            k += 1

    return images, x, y


if __name__ == "__main__":
    path = "data/binaryalphadigs.mat"
    data, x_im, y_im = lire_alpha_digit(path, [0, 1])
    n, p = data.shape

    # Variables
    config = (p, 100, 50, 100)
    epochs_rbm = 100
    epochs_dnn = 200
    learning_rate = 0.1
    batch_size = 100
