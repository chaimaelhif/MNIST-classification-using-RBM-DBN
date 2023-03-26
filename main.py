import argparse

import numpy as np
from sklearn.utils import shuffle

from export_import import lire_mnist, lire_alpha_digit, save_model, import_model
from plots import plot_loss
from principal_DBN_alpha import DBN
from principal_DNN_MNIST import DNN
from principal_RBM_alpha import RBM
from principal_VAE import VAE

X_MNIST = 28
Y_MNIST = 28

X_BAD = 20
Y_BAD = 16


if __name__ == "__main__":
    # Define arguments
    parser = argparse.ArgumentParser(description='Description of the program')
    parser.add_argument('--action',
                        choices=["RBM", "DBN", "DNN5.1", "DNN5.2.1", "DNN5.2.2", "VAE"],
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
    epochs_dnn = 100
    learning_rate = 0.08
    batch_size = 128
    nb_gibbs = 500
    digits = np.arange(0, output_dim)

    # Import dataset: Binary Alpha Digits (BAD)
    path = "data/binaryalphadigs.mat"
    bad = lire_alpha_digit(path, [5])
    n_bad, p_bad = bad.shape

    # Import dataset: MNIST
    path = "data/mnist_all.mat"
    mnist_train, label_train = lire_mnist(path, digits, "train")
    mnist_test, label_test = lire_mnist(path, digits, "test")
    n_mnist, p_mnist = mnist_train.shape

    if args.action == "RBM":
        # Test RBM
        q = 200  # number of hidden values
        rbm = RBM(p_bad, q)
        rbm.train_rbm(bad, epochs_rbm, learning_rate, batch_size)
        rbm.generate_image_rbm(X_BAD, Y_BAD, 4, nb_gibbs, True)

    elif args.action == "DBN":
        # Test DBN
        config = (p_bad, 200, 200)
        dbn = DBN(config)
        dbn.train_dbn(bad, epochs_dnn, learning_rate, batch_size)
        dbn.generate_image_dbn(X_BAD, Y_BAD, 4, nb_gibbs, True)

        if args.arg1 == "test":
            nb_layers = [2, 5, 10, 15]
            for layer in nb_layers:
                # Define the config for each iteration
                config = [p_bad]
                config.extend([200 for x in range(layer)])

                dbn = DBN(config)
                dbn.train_dbn(bad, epochs_dnn, learning_rate, batch_size)
                dbn.generate_image_dbn(X_BAD, Y_BAD, 4, nb_gibbs, True)

            nb_neurons = [10, 100, 200, 300]
            for neurons in nb_neurons:
                # Define the config for each iteration
                config = [p_bad]
                config.extend([neurons for x in range(2)])

                dbn = DBN(config)
                dbn.train_dbn(bad, epochs_dnn, learning_rate, batch_size)
                dbn.generate_image_dbn(X_BAD, Y_BAD, 4, nb_gibbs, True)

    elif args.action == "VAE":
        # Variables to VAE
        n_rows = X_BAD
        n_cols = Y_BAD
        n_channels = 1
        n_pixels = n_rows * n_cols
        img_shape = (n_rows, n_cols, n_channels)
        z_dim = 10
        vae_dim_1 = 512
        vae_dim_2 = 256
        n_epochs = 200

        # Test VAE
        config = {
            "x_dim": n_pixels,
            "h_dim1": vae_dim_1,
            "h_dim2": vae_dim_2,
            "z_dim": z_dim,
            "n_rows": n_rows,
            "n_cols": n_cols,
            "n_channels": n_channels
        }
        vae = VAE(**config)
        vae.train_vae(bad, epochs=n_epochs)

        # Save model
        path = "VAE"
        save_model(path, vae)

        vae.generate_images_vae(4)

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
            path = f"dnn_testPretrained{dnn.pretrained}.pkl"
            save_model(path, dnn)

        elif args.arg1 == "test":
            # Import model
            path = "dnn_test.pkl"
            dnn = import_model(path)

        # Test model
        print(f"Model DNN with pretrained={dnn.pretrained}:")
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
            path_1 = f"dnn_testPretrained{dnn1.pretrained}.pkl"
            path_2 = f"dnn_testPretrained{dnn2.pretrained}.pkl"
            save_model(path_1, dnn1)
            save_model(path_2, dnn2)

        elif args.arg1 == "test":
            # Import models
            path_1 = "dnn_testPretrainedTrue.pkl"
            dnn1 = import_model(path_1)
            path_2 = "dnn_testPretrainedFalse.pkl"
            dnn2 = import_model(path_2)

        # 5
        print(f"Model DNN with pretrained={dnn1.pretrained} for Train dataset:")
        dnn1.test_dnn(mnist_train, label_train)

        print(f"Model DNN with pretrained={dnn1.pretrained} for Test dataset:")
        dnn1.test_dnn(mnist_test, label_test)

        print(f"Model DNN with pretrained={dnn2.pretrained} for Train dataset:")
        dnn2.test_dnn(mnist_train, label_train)

        print(f"Model DNN with pretrained={dnn2.pretrained} for Test dataset:")
        dnn2.test_dnn(mnist_test, label_test)

    elif args.action == "DNN5.2.2":
        # 6
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
                    path_1 = f"dnn_testPretrained{dnn1.pretrained}_NbLayers{str(layer)}.pkl"
                    path_2 = f"dnn_testPretrained{dnn2.pretrained}_NbLayers{str(layer)}.pkl"
                    save_model(path_1, dnn1)
                    save_model(path_2, dnn2)

                elif args.arg1 == "test":
                    # Import models
                    path_1 = f"dnn_testPretrainedTrue_NbLayers{str(layer)}.pkl"
                    path_2 = f"dnn_testPretrainedFalse_NbLayers{str(layer)}.pkl"
                    dnn1 = import_model(path_1)
                    dnn2 = import_model(path_2)

                # Compute error rate for both models
                print(f"Model DNN with pretrained={dnn1.pretrained} for nb_layer= {layer}:")
                error1 = dnn1.test_dnn(mnist_test, label_test)

                print(f"Model DNN with pretrained={dnn2.pretrained} for nb_layer= {layer}:")
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
                config.extend([neurons for x in range(2)])

                if args.arg1 == "train":
                    dnn1 = DNN(config)  # to pretrain
                    dnn2 = DNN(config)

                    # Pretrain model then train it
                    dnn1.pretrain_dnn(mnist_train, epochs_rbm, learning_rate, batch_size)
                    dnn1.backward_propagation(mnist_train, label_train, epochs_dnn, learning_rate, batch_size)

                    # Train model directly
                    dnn2.backward_propagation(mnist_train, label_train, epochs_dnn, learning_rate, batch_size)

                    # Save both models
                    path_1 = f"dnn_testPretrained{dnn1.pretrained}_NbNeurons{str(neurons)}.pkl"
                    path_2 = f"dnn_testPretrained{dnn2.pretrained}_NbNeurons{str(neurons)}.pkl"
                    save_model(path_1, dnn1)
                    save_model(path_2, dnn2)

                elif args.arg1 == "test":
                    # Import models
                    path_1 = f"dnn_testPretrainedTrue_NbNeurons{str(neurons)}.pkl"
                    path_2 = f"dnn_testPretrainedFalse_NbNeurons{str(neurons)}.pkl"
                    dnn1 = import_model(path_1)
                    dnn2 = import_model(path_2)

                # Compute error rate for both models
                print(f"Model DNN with pretrained={dnn1.pretrained} for nb_neurons= {neurons}:")
                error1 = dnn1.test_dnn(mnist_test, label_test)

                print(f"Model DNN with pretrained={dnn2.pretrained} for nb_neurons= {neurons}:")
                error2 = dnn2.test_dnn(mnist_test, label_test)

                error_rate1.append(error1)
                error_rate2.append(error2)

            plot_loss(error_rate1, error_rate2, nb_neurons, "Number of neurons")

        if args.arg3 == "train_size":  # Test with different size of training data
            error_rate1, error_rate2 = [], []
            train_size = [1000, 3000, 7000, 10000, 30000, 60000]
            for size in train_size:
                config = (p_mnist, 200, 200)

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
                    path_1 = f"dnn_testPretrained{dnn1.pretrained}_TrainSize{str(size)}.pkl"
                    path_2 = f"dnn_testPretrained{dnn2.pretrained}_TrainSize{str(size)}.pkl"
                    save_model(path_1, dnn1)
                    save_model(path_2, dnn2)

                elif args.arg1 == "test":
                    # Import models
                    path_1 = f"dnn_testPretrainedTrue_TrainSize{str(size)}.pkl"
                    path_2 = f"dnn_testPretrainedFalse_TrainSize{str(size)}.pkl"
                    dnn1 = import_model(path_1)
                    dnn2 = import_model(path_2)

                # Compute error rate for both models
                print(f"Model DNN with pretrained={dnn1.pretrained} for train_size= {size}:")
                error1 = dnn1.test_dnn(mnist_test, label_test)

                print(f"Model DNN with pretrained={dnn2.pretrained} for train_size= {size}:")
                error2 = dnn2.test_dnn(mnist_test, label_test)

                error_rate1.append(error1)
                error_rate2.append(error2)

            plot_loss(error_rate1, error_rate2, train_size, "Sample size")
