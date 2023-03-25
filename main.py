import pickle
import numpy as np
import scipy as sp

from principal_RBM_alpha import RBM
from principal_DBN_alpha import DBN
from principal_DNN_MNIST import DNN

X_MNIST = 28
Y_MNIST = 28

X_BAD = 20
Y_BAD = 16


def lire_alpha_digit(filename: str, indices: np.ndarray = None):
    mat = sp.io.loadmat(filename, simplify_cells=True)
    bad = mat["dat"][indices, :]
    images = np.zeros((bad.size, bad[0, 0].size))
    k = 0  # image index
    for i in range(bad.shape[0]):
        for j in range(bad.shape[1]):
            images[k, :] = bad[i, j].flatten()
            k += 1

    return images


def lire_mnist(filename: str, indices: np.ndarray, data_type: str):
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


if __name__ == "__main__":
    # Global Variables
    output_dim = 10
    epochs_rbm = 100
    epochs_dnn = 200
    learning_rate = 0.1
    batch_size = 100
    nb_gibbs = 500
    digits = np.arange(0, output_dim)

    # Binary Alpha Digits (BAD)
    path = "data/binaryalphadigs.mat"
    data = lire_alpha_digit(path, digits)
    n_bad, p_bad = data.shape

    # MNIST
    path = "data/mnist_all.mat"
    mnist_train, label_train = lire_mnist(path, digits, "train")
    mnist_test, label_test = lire_mnist(path, digits, "test")
    n_mnist, p_mnist = mnist_train.shape

    args = "DNN"

    if args == "RBM":
        # Test RBM
        q = 200  # number of hidden values
        config = (p_bad, 100, 100, 50)
        rbm = RBM(p_bad, q)
        rbm.train_rbm(data, epochs_rbm, learning_rate, batch_size)
        rbm.generate_image_rbm(X_BAD, Y_BAD, 4, nb_gibbs, True)

    elif args == "DBN":
        # Test DBN
        config = (p_bad, 100, 100, 50)
        dbn = DBN(config)
        dbn.train_dbn(data, epochs_dnn, learning_rate, batch_size)
        dbn.generate_image_dbn(X_BAD, Y_BAD, 4, nb_gibbs, True)

    elif args == "DNN":
        # Test DNN (5.1)
        config = (p_mnist, 100, 100, 50)
        dnn = DNN(config)

        dnn.pretrain_dnn(mnist_train, epochs_rbm, learning_rate, batch_size)
        dnn.backward_propagation(
            mnist_train, label_train, epochs_dnn, learning_rate, batch_size
        )
        dnn.test_dnn(mnist_test, label_test)
        idx = np.where(label_test == 1)[0]

        # plot probabilities of the first class
        dnn.plot_proba(mnist_test.iloc[idx, :])

        # save model
        with open("dnn_test.pkl", "wb") as f:
            pickle.dump(dnn, f)

        # Test 2 DNN := pretrained VS not pretrained (5.2)
        # 1
        dnn1 = DNN(config)  # to pretrain
        dnn2 = DNN(config)

        # 2
        dnn1.pretrain_dnn(mnist_train, epochs_rbm, learning_rate, batch_size)

        # 3
        dnn1.backward_propagation(mnist_train, label_train, epochs_dnn, learning_rate, batch_size)

        # 4
        dnn2.backward_propagation(mnist_train, label_train, epochs_dnn, learning_rate, batch_size)

        # 5
        print("DNN with pretraining:")
        print("For Train dataset:")
        dnn1.test_dnn(mnist_train, label_train)

        print("For Test dataset:")
        dnn1.test_dnn(mnist_test, label_test)

        print("DNN without pretraining:")
        print("For Train dataset:")
        dnn2.test_dnn(mnist_train, label_train)

        print("For Test dataset:")
        dnn2.test_dnn(mnist_test, label_test)
