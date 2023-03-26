import argparse
import pickle

import numpy as np
import scipy as sp
from sklearn.utils import shuffle

from plots import plot_loss
from principal_DBN_alpha import DBN
from principal_DNN_MNIST import DNN
from principal_RBM_alpha import RBM

X_MNIST = 28
Y_MNIST = 28

X_BAD = 20
Y_BAD = 16


def lire_alpha_digit(filename: str, indices=None):
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
    parser = argparse.ArgumentParser(description='Description of the program')
    parser.add_argument('--action',
                        choices=["RBM", "DBN", "DNN5.1", "DNN5.2.1", "DNN5.2.2"],
                        help='Generate images using RBM')
    parser.add_argument('--arg1', choices=["train", "test"], help='To train or test model')
    parser.add_argument('--arg2', help='Boolean to specify pretraining')
    parser.add_argument('--arg3',
                        choices=["nb_layers", "nb_neurons", "train_size"],
                        help='To specify the type of analysis to make')
    args = parser.parse_args()

    # Global Variables
    output_dim = 10
    epochs_rbm = 100
    epochs_dnn = 200
    learning_rate = 0.08
    batch_size = 128
    nb_gibbs = 500
    digits = np.arange(0, output_dim)

    # Import dataset: Binary Alpha Digits (BAD)
    path = "data/binaryalphadigs.mat"
    data = lire_alpha_digit(path, [5])
    n_bad, p_bad = data.shape

    # Import dataset: MNIST
    path = "data/mnist_all.mat"
    mnist_train, label_train = lire_mnist(path, digits, "train")
    mnist_test, label_test = lire_mnist(path, digits, "test")
    n_mnist, p_mnist = mnist_train.shape

    if args.action == "RBM":
        # Test RBM
        q = 200  # number of hidden values
        rbm = RBM(p_bad, q)
        rbm.train_rbm(data, epochs_rbm, learning_rate, batch_size)
        rbm.generate_image_rbm(X_BAD, Y_BAD, 4, nb_gibbs, True)

    elif args.action == "DBN":
        # Test DBN
        config = (p_bad, 200, 200)
        dbn = DBN(config)
        dbn.train_dbn(data, epochs_dnn, learning_rate, batch_size)
        dbn.generate_image_dbn(X_BAD, Y_BAD, 4, nb_gibbs, True)

    elif args.action == "DNN5.1":
        # Test DNN (5.1)
        config = (p_mnist, 200, 200)

        if args.arg1 == "train":
            dnn = DNN(config)
            if args.arg2:  # Train model (with pretraining)
                dnn.pretrain_dnn(mnist_train, epochs_rbm, learning_rate, batch_size)

            dnn.backward_propagation(
                mnist_train, label_train, epochs_dnn, learning_rate, batch_size
            )

            # Save model
            with open(f"dnn_testPretrained{dnn.pretrained}.pkl", "wb") as f:
                pickle.dump(dnn, f)

        elif args.arg1 == "test":
            # Import model
            with open("dnn_test.pkl", "rb") as f:
                dnn = pickle.load(f)

        # Test model
        dnn.test_dnn(mnist_test, label_test)

        # Plot probabilities of the class k
        k = 2  # Choose a class and plot the probability
        idx = np.where(label_test == k)[0]
        dnn.plot_proba(mnist_test[idx])

    elif args.action == "DNN5.2.1":
        # Test 2 DNN := pretrained VS not pretrained (5.2)
        config = (p_mnist, 200, 200)

        if args.arg1 == "train":
            # 1
            dnn1 = DNN(config)  # to pretrain
            dnn2 = DNN(config)

            # 2
            dnn1.pretrain_dnn(mnist_train, epochs_rbm, learning_rate, batch_size)

            # 3
            dnn1.backward_propagation(mnist_train, label_train, epochs_dnn, learning_rate, batch_size)

            # 4
            dnn2.backward_propagation(mnist_train, label_train, epochs_dnn, learning_rate, batch_size)

            # Save both models
            with open(f"dnn_testPretrained{dnn1.pretrained}.pkl", "wb") as f:
                pickle.dump(dnn1, f)  # pretrained
            with open(f"dnn_testPretrained{dnn2.pretrained}.pkl", "wb") as f:
                pickle.dump(dnn2, f)

        elif args.arg1 == "test":
            # Import models
            with open("dnn_testPretrainedTrue.pkl", "rb") as f:
                dnn1 = pickle.load(f)

            with open("dnn_testPretrainedFalse.pkl", "rb") as f:
                dnn2 = pickle.load(f)

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

    # 6
    elif args.action == "DNN5.2.2":
        if args.arg3 == "nb_layers":  # Test with different number layers
            error_rate1, error_rate2 = [], []
            nb_layers = [2, 3, 5, 7]
            for layer in nb_layers:
                # Define the config for each iteration
                config = [p_mnist]
                config.extend([200 for x in range(layer)])

                if args.arg1 == "train":
                    dnn1 = DNN(config)  # to pretrain
                    dnn2 = DNN(config)

                    # Pretrain model then train it
                    dnn1.pretrain_dnn(mnist_train, epochs_rbm, learning_rate, batch_size)
                    dnn1.backward_propagation(mnist_train, label_train, epochs_dnn, learning_rate, batch_size)

                    # Train model directly
                    dnn2.backward_propagation(mnist_train, label_train, epochs_dnn, learning_rate, batch_size)

                    # Save both models
                    with open(f"dnn_testPretrained{dnn1.pretrained}_NbLayers{str(layer)}.pkl", "wb") as f:
                        pickle.dump(dnn1, f)  # pretrained
                    with open(f"dnn_testPretrained{dnn2.pretrained}_NbLayers{str(layer)}.pkl", "wb") as f:
                        pickle.dump(dnn2, f)

                elif args.arg1 == "test":
                    # Import models
                    with open(f"dnn_testPretrainedTrue_NbLayers{str(layer)}.pkl", "rb") as f:
                        dnn1 = pickle.load(f)
                    with open(f"dnn_testPretrainedFalse_NbLayers{str(layer)}.pkl", "rb") as f:
                        dnn2 = pickle.load(f)

                # Compute error rate for both models
                error1 = dnn1.test_dnn(mnist_test, label_test)
                error2 = dnn2.test_dnn(mnist_test, label_test)

                error_rate1.append(error1)
                error_rate2.append(error2)

            plot_loss(error_rate1, error_rate2, nb_layers, "Number of layers")

            if args.arg3 == "nb_neurons":  # Test with different number neurons
                error_rate1, error_rate2 = [], []
                nb_neurons = [100, 200, 300, 700]
                for neurons in nb_neurons:
                    # Define the config for each iteration
                    config = [p_mnist]
                    config.extend([neurons for x in range(3)])

                    if args.arg1 == "train":
                        dnn1 = DNN(config)  # to pretrain
                        dnn2 = DNN(config)

                        # Pretrain model then train it
                        dnn1.pretrain_dnn(mnist_train, epochs_rbm, learning_rate, batch_size)
                        dnn1.backward_propagation(mnist_train, label_train, epochs_dnn, learning_rate, batch_size)

                        # Train model directly
                        dnn2.backward_propagation(mnist_train, label_train, epochs_dnn, learning_rate, batch_size)

                        # Save both models
                        with open(f"dnn_testPretrained{dnn1.pretrained}_NbNeurons{str(neurons)}.pkl", "wb") as f:
                            pickle.dump(dnn1, f)  # pretrained
                        with open(f"dnn_testPretrained{dnn2.pretrained}_NbNeuron{str(neurons)}.pkl", "wb") as f:
                            pickle.dump(dnn2, f)
                    elif args.arg1 == "test":
                        # Import models
                        with open(f"dnn_testPretrainedTrue_NbNeurons{str(neurons)}.pkl", "rb") as f:
                            dnn1 = pickle.load(f)
                        with open(f"dnn_testPretrainedFalse_NbNeurons{str(neurons)}.pkl", "rb") as f:
                            dnn2 = pickle.load(f)

                    # Compute error rate for both models
                    error1 = dnn1.test_dnn(mnist_test, label_test)
                    error2 = dnn2.test_dnn(mnist_test, label_test)

                    error_rate1.append(error1)
                    error_rate2.append(error2)

                plot_loss(error_rate1, error_rate2, nb_neurons, "Number of neurons")

        if args.arg3 == "nb_layers":  # Test with different size of training data
            error_rate1, error_rate2 = [], []
            train_size = [1000, 3000, 7000, 10000, 30000, 60000]
            for size in train_size:
                config = (p_mnist, 100, 100, 50)

                if args.arg1 == "train":
                    dnn1 = DNN(config)  # to pretrain
                    dnn2 = DNN(config)

                    # Select the training data for each iteration
                    data_shuffled, labels_shuffled = shuffle(mnist_train, label_train)
                    data_sampled, labels_sampled = data_shuffled[:size], labels_shuffled[:size]

                    # Pretrain model then train it
                    dnn1.pretrain_dnn(data_sampled, epochs_rbm, learning_rate, batch_size)
                    dnn1.backward_propagation(data_sampled, labels_sampled, epochs_dnn, learning_rate, batch_size)

                    # Train model directly
                    dnn2.backward_propagation(data_sampled, labels_sampled, epochs_dnn, learning_rate, batch_size)

                    # Save both models
                    with open(f"dnn_testPretrained{dnn1.pretrained}_TrainSize{str(size)}.pkl", "wb") as f:
                        pickle.dump(dnn1, f)  # pretrained
                    with open(f"dnn_testPretrained{dnn2.pretrained}_TrainSize{str(size)}.pkl", "wb") as f:
                        pickle.dump(dnn2, f)
                elif args.arg1 == "test":
                    # Import models
                    with open(f"dnn_testPretrainedTrue_TrainSize{str(size)}.pkl", "rb") as f:
                        dnn1 = pickle.load(f)
                    with open(f"dnn_testPretrainedFalse_TrainSize{str(size)}.pkl", "rb") as f:
                        dnn2 = pickle.load(f)

                # Compute error rate for both models
                error1 = dnn1.test_dnn(mnist_test, label_test)
                error2 = dnn2.test_dnn(mnist_test, label_test)

                error_rate1.append(error1)
                error_rate2.append(error2)

            plot_loss(error_rate1, error_rate2, train_size, "Sample size")
