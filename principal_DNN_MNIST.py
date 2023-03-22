import numpy as np
from sklearn.utils import shuffle

from principal_DBN_alpha import DBN
from principal_RBM_alpha import RBM, sigmoid

global n, p, x_im, y_im


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def calcul_softmax(rbm: RBM, data):
    z = rbm.entree_sortie_rbm(data)
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)


class DNN:
    def __init__(self, config=(p, 100, 50, 100, 10)):
        self.config: tuple = config
        self.num_layers: int = len(self.config)

        # un DNN est un DBN avec une couche de classification supplémentaire
        # dernier RBM du DBN pour la classification → on ne définit pas "rbm.a".
        self.dnn = DBN(config)

    def pretrain_dnn(
            self, data, epochs=100, learning_rate=0.1, batch_size=100
    ):
        self.dnn.train_dbn(data, epochs, learning_rate, batch_size)
        return self

    def entree_sortie_network(self, data):
        v = data.copy()
        results = [v]  # Couche d'entrée
        for i in range(self.num_layers - 1):
            v = self.dnn.dbn[i].entree_sortie_rbm(v)
            results.append(v)

        # Compute the probabilities
        softmax_probas = calcul_softmax(self.dnn.dbn[self.num_layers - 1], v)
        results.append(softmax_probas)
        return results

    def backward_propagation(
        self, data, labels, epochs=100, learning_rate=0.1, batch_size=100
    ):
        train_acc, train_loss, test_acc, test_loss = 0, 0, 0, 0

        for epoch in range(epochs):
            data_copy = data.copy()
            labels_copy = labels.copy()
            data_copy, labels_copy = shuffle(data_copy, labels_copy)

            for batch in range(0, data.shape[0], batch_size):
                data_batch = data_copy[
                    batch: min(batch + batch_size, data.shape[0]), :
                ]
                labels_batch = labels_copy[
                    batch: min(batch + batch_size, data.shape[0]), :
                ]
                # Forward pass
                activations = self.entree_sortie_network(data_batch)

                # Backward pass
                # Start with last layer
                delta = activations[-1] - labels_batch
                grad_w = np.dot(activations[-2].T, delta)
                grad_b = np.sum(delta, axis=0)
                self.dnn.dbn[-1].W -= learning_rate * grad_w
                self.dnn.dbn[-1].b -= learning_rate * grad_b

                # Propagate error backwards through hidden layers
                for layer in range(2, self.num_layers):
                    delta = np.dot(delta,
                                   self.dnn.dbn[-layer + 1].W.T
                                   ) * sigmoid_prime(activations[-layer])
                    grad_w = np.dot(activations[-layer - 1].T, delta)
                    grad_b = np.sum(delta, axis=0)
                    self.dnn.dbn[-layer].W -= learning_rate * grad_w
                    self.dnn.dbn[-layer].b -= learning_rate * grad_b

                # Compute training accuracy and cross-entropy loss
                train_acc = np.mean(
                    np.argmax(activations[-1], axis=1) == labels_batch
                )
                train_loss = -np.mean(
                    np.sum(labels_batch * np.log(activations[-1]), axis=1)
                )

            # Compute test accuracy and cross-entropy loss
            test_activations = self.entree_sortie_network(data)
            test_acc = np.mean(
                np.argmax(test_activations[-1], axis=1) == labels
            )
            test_loss = -np.mean(
                np.sum(labels * np.log(test_activations[-1]), axis=1)
            )

            # Print progress
            print(
                f"Epoch {epoch}/{epochs}: "
                f"train loss = {train_loss:.4f}, "
                f"train accuracy = {train_acc:.4f}, "
                f"test loss = {test_loss:.4f}, "
                f"test accuracy = {test_acc:.4f}"
            )

        return self

    def test_dnn(self, test_data, test_labels):
        num_correct = 0
        # Loop through each test data point and its corresponding label
        for i, (data, label) in enumerate(zip(test_data, test_labels)):
            # Compute the predicted label using the DNN
            probs = self.entree_sortie_network(data)
            pred_label = np.argmax(probs[-1])

            # Update the number of correctly classified data points
            if pred_label == label:
                num_correct += 1

        # Print the error rate and return it
        error_rate = num_correct / test_data.shape[0]
        print(f"Test error rate: {error_rate:.2%}")
        return error_rate
