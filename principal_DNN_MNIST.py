import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from principal_DBN_alpha import DBN
from principal_RBM_alpha import RBM


def sigmoid_prime(z):
    return z * (1 - z)


def calcul_softmax(rbm: RBM, data):
    vdn = np.dot(data, rbm.W)
    z = rbm.b + np.dot(data, rbm.W)
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)


class DNN:
    def __init__(self, config):
        self.config: tuple = config
        self.num_layers: int = len(self.config)

        # un DNN est un DBN avec une couche de classification supplémentaire
        # dernier RBM du DBN pour la classification → on ne définit pas "rbm.a".
        self.dbn = DBN(config)
        self.classification = RBM(config[-1], 10)

    def pretrain_dnn(
            self, data, epochs=100, learning_rate=0.1, batch_size=100
    ):
        self.dbn.train_dbn(data, epochs, learning_rate, batch_size)
        return self

    def entree_sortie_network(self, data):
        v = data.copy()
        results = [v]  # Couche d'entrée
        for i in range(self.num_layers-1):
            p_h = self.dbn.dbn[i].entree_sortie_rbm(v)
            v = np.random.binomial(1, p_h)
            results.append(v)

        # Compute the probabilities
        softmax_probas = calcul_softmax(self.classification, v)
        results.append(softmax_probas)
        return results

    def backward_propagation(
        self, data, labels, epochs=100, learning_rate=0.1, batch_size=100
    ):
        train_acc, train_loss, test_acc, test_loss = 0, 0, 0, 0

        for epoch in range(epochs):
            data_copy = data.copy()
            labels_copy = pd.get_dummies(labels.copy())
            data_copy, labels_copy = shuffle(data_copy, labels_copy)

            for batch in range(0, data.shape[0], batch_size):
                data_batch = data_copy[
                    batch: min(batch + batch_size, data.shape[0]), :
                ]
                labels_batch = labels_copy[
                    batch: min(batch + batch_size, data.shape[0])
                ]
                # Forward pass
                activations = self.entree_sortie_network(data_batch)

                # Backward pass
                # Start with last layer
                delta = activations[-1] - labels_batch
                grad_w = np.dot(activations[-2].T, delta)/batch_size
                grad_b = np.mean(delta, axis=0)
                self.classification.W -= learning_rate * grad_w
                self.classification.b -= learning_rate * grad_b

                # Propagate error backwards through hidden layers
                for layer in range(1, self.num_layers):
                    if layer == 1:
                        delta = np.dot(delta,
                                       self.classification.W.T
                                       ) * sigmoid_prime(activations[-layer-1])
                    else:
                        delta = np.dot(delta,
                                       self.dbn.dbn[-layer + 1].W.T
                                       ) * sigmoid_prime(activations[-layer-1])
                    grad_w = np.dot(activations[-layer - 2].T, delta)/batch_size
                    grad_b = np.mean(delta, axis=0)
                    self.dbn.dbn[-layer].W -= learning_rate * grad_w
                    self.dbn.dbn[-layer].b -= learning_rate * grad_b

                # Compute training accuracy and cross-entropy loss
                c = pd.from_dummies(labels_batch)
                train_acc = np.mean(
                    np.argmax(activations[-1], axis=1).reshape(-1,1) == c
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
                np.sum(labels_copy * np.log(test_activations[-1]), axis=1)
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
