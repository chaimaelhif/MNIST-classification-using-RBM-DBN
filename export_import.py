import pickle

import numpy as np
import scipy as sp

from principal_DNN_MNIST import DNN


def lire_alpha_digit(filename: str, indices=None):
    """
    :param filename: path to import data
    :param indices: index of alpha digits we want to use for training
    :return: array (n, p): n number of sample, p number of pixels
    """
    mat = sp.io.loadmat(filename, simplify_cells=True)
    bad = mat["dat"][indices, :]
    images = np.zeros((bad.size, bad[0, 0].size))
    im = 0  # image index
    for i in range(bad.shape[0]):
        for j in range(bad.shape[1]):
            images[im, :] = bad[i, j].flatten()
            im += 1

    return images


def lire_mnist(filename: str, indices: np.ndarray, data_type: str):
    """
    :param filename: path to import data
    :param indices: index of alpha digits we want to use for training
    :param data_type: "train", "test"
    :return: array (n, p): n number of sample, p number of pixels
    """
    mnist_all = sp.io.loadmat(filename, simplify_cells=True)
    key = data_type + "0"
    data_mnist = (mnist_all[key] > 127).astype(int)
    label = np.zeros(mnist_all[key].shape[0])
    for i in indices[1:]:
        key = data_type + str(i)
        data_mnist = np.vstack([data_mnist, (mnist_all[key] > 127).astype(int)])
        y = i * np.ones(mnist_all[key].shape[0])
        label = np.concatenate([label, y], axis=0)
    return data_mnist, label


def import_model(filename: str) -> DNN:
    """
    :param filename: file path
    :return: trained model
    """
    with open("models/"+filename, "rb") as file:
        model = pickle.load(file)
        return model


def save_model(filename: str, model):
    """
    :param filename: file path
    :param model: trained model to save
    """
    with open("models/"+filename, "wb") as file:
        pickle.dump(model, file)
