import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from principal_DBN_alpha import DBN
from principal_RBM_alpha import RBM


def sigmoid_prime(z):
    return z * (1 - z)


def calcul_softmax(rbm: RBM, data):
    z = np.array(rbm.b) + np.dot(data, rbm.W)
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)


class DNN:
    def __init__(self, config, output_dim=10, pretrain=False):
        self.config: tuple = config
        self.num_layers: int = len(self.config)

        # un DNN est un DBN avec une couche de classification supplémentaire
        # dernier RBM du DBN pour la classification → on ne définit pas "rbm.a".
        self.dbn: DBN = DBN(config)
        self.classification: RBM = RBM(config[-1], output_dim)
        self.pretrained: bool = pretrain
        self.fitted: bool = False

    def pretrain_dnn(self, data, epochs=100, learning_rate=0.1, batch_size=100):
        self.dbn.train_dbn(data, epochs, learning_rate, batch_size)
        self.pretrained = True
        return self

    def entree_sortie_network(self, data):
        v = data.copy()
        results = [v]  # Couche d'entrée
        for i in range(self.num_layers - 1):
            p_h = self.dbn.dbn[i].entree_sortie_rbm(v)
            v = np.random.binomial(1, p_h)
            results.append(p_h)

        # Compute the probabilities
        softmax_probas = calcul_softmax(self.classification, v)
        results.append(softmax_probas)
        return results

    def backward_propagation(
        self,
        data,
        labels,
        epochs=100,
        learning_rate=0.1,
        batch_size=100,
        verbose=True,
        plot=True,
    ):
        loss_batches, loss = [], []

        for epoch in range(epochs):
            data_copy = data.copy()
            labels_copy = pd.get_dummies(labels.copy())
            data_copy, labels_copy = shuffle(data_copy, labels_copy)

            for batch in range(0, data.shape[0], batch_size):
                data_batch = data_copy[
                    batch : min(batch + batch_size, data.shape[0]), :
                ]
                labels_batch = labels_copy[
                    batch : min(batch + batch_size, data.shape[0])
                ]
                # Forward pass
                activations = self.entree_sortie_network(data_batch)

                # Loss
                loss_batches.append(
                    -np.mean(np.sum(labels_batch * np.log(activations[-1]), axis=1))
                )

                # Backward pass
                # Start with last layer
                delta = activations[-1] - labels_batch
                grad_w = np.dot(activations[-2].T, delta) / batch_size
                grad_b = np.mean(delta, axis=0)
                self.classification.W -= learning_rate * grad_w
                self.classification.b -= learning_rate * grad_b

                # Propagate error backwards through hidden layers
                for layer in range(1, self.num_layers):
                    if layer == 1:
                        delta = np.dot(delta, self.classification.W.T) * sigmoid_prime(
                            activations[-layer - 1]
                        )
                    else:
                        delta = np.dot(
                            delta, self.dbn.dbn[-layer + 1].W.T
                        ) * sigmoid_prime(activations[-layer - 1])
                    if layer == self.num_layers - 1:
                        grad_w = np.dot(data_batch.T, delta) / batch_size
                    else:
                        grad_w = np.dot(activations[-layer - 2].T, delta) / batch_size
                    grad_b = np.mean(delta, axis=0)
                    self.dbn.dbn[-layer].W -= learning_rate * grad_w
                    self.dbn.dbn[-layer].b -= learning_rate * grad_b

            # Compute cross-entropy loss
            train_loss = np.mean(loss_batches)
            loss.append(train_loss)
            # Print progress
            if verbose:
                print(
                    f"Epoch {epoch}/{epochs}: Train error -----------------{train_loss:.4f}"
                )
        if plot:
            plt.plot(np.arange(epochs), loss)
            plt.xlabel("Epochs")
            plt.ylabel("CrossEntropy Loss")
            if self.pretrained:
                plt.title("Loss for pretrained DNN")
            else:
                plt.title("Loss for DNN (without pretraining)")
            plt.show()

        self.fitted = True
        return self

    def test_dnn(self, test_data, test_labels, verbose=True):
        probs = self.entree_sortie_network(test_data)
        pred_label = np.argmax(probs[-1], axis=1)
        num_correct = np.sum(test_labels != pred_label)

        # Print the error rate and return it
        error_rate = num_correct / test_data.shape[0]

        if verbose:
            print(f"Error rate ----------------- : {error_rate:.2%}")
        return error_rate

    def plot_proba(self, data):
        pred_labels = self.entree_sortie_network(data)[-1]
        plt.plot(pred_labels)
        plt.legend()
        plt.xlabel("Classes")
        plt.ylabel("Predicted probability for each class")
        plt.title("")
        plt.show()
