import numpy as np

from math_helpers import sigmoid, sigmoid_derivative


class NeuralNetwork:
    def __init__(self, layer_sizes):

        print(f"Initializing network: {str(layer_sizes)}")

        self.layer_count = len(layer_sizes)

        self.layer_sizes = layer_sizes

        # Create weight matrices for each layer
        self.weights = [np.random.randn(j, k) for k, j in zip(layer_sizes[:-1], layer_sizes[1:])]

        # Create the biases for layer_sizes[1:] (except input layer)
        self.biases = [np.random.randn(n, 1) for n in layer_sizes[1:]]

    def train_SGD(self, x_train, y_train, epochs, mini_batch_size, learning_rate, test_data=None):
        training_data = (x_train, y_train)
        n = x_train.shape[1]

        for j in range(epochs):
            np.random.shuffle(list(training_data))

            mini_batches_x = [x_train.T[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            mini_batches_y = [y_train.T[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]

            for batch in zip(mini_batches_x, mini_batches_y):
                self.update(batch, learning_rate)

            if test_data is not None:
                print(f"Epoch {j}: {self.evaluate(test_data)} / {test_data[0].shape[1]}")

            else:
                print(f"Epoch {j} complete")

    def update(self, batch, learning_rate):
        """
        Updates the network weights and biases by evaluating these mini batches with backpropagation
        :param batch:
        :param learning_rate:
        """
        nb_b = [np.zeros(b.shape) for b in self.biases]
        nb_w = [np.zeros(w.shape) for w in self.weights]

        # for input and label y in batch

        for i in range(batch[0].shape[0]):
            # gets the cost gradient splitted up in b and w components
            d_nb_b, d_nb_w = self.backpropagate(batch[0][i].reshape(784, 1), batch[1][i].reshape(10, 1))
            nb_b = [nb + dnb for nb, dnb in zip(nb_b, d_nb_b)]
            nb_w = [nw + dnw for nw, dnw in zip(nb_w, d_nb_w)]

        self.weights = [w - (learning_rate / len(batch)) * nw for w, nw in zip(self.weights, nb_w)]
        self.biases = [b - (learning_rate / len(batch)) * nb for b, nb in zip(self.biases, nb_b)]

    def evaluate(self, test_data):
        """
        Return the output of the current Network if the input is training
        :param test_data: array of input data
        :return: the computed output vector of the current network
        """

        test_results = [(np.argmax(self.compute(x)), np.argmax(y)) for x, y in zip(test_data[0].T, test_data[1].T)]
        return sum(int(x == y) for x, y in test_results)

    def compute(self, a):
        """
        Evaluate the network for input a
        :param a:
        :return: network output for input a
        """
        for b, w in zip(self.biases, self.weights):
            tmp = np.dot(w, a).reshape(w.shape[0], 1)
            a = sigmoid(tmp + b)
        return a

    def backpropagate(self, input_data, expected_data):
        # Compute cost
        nb_b = [np.zeros(b.shape) for b in self.biases]
        nb_w = [np.zeros(w.shape) for w in self.weights]

        a = input_data
        activations = [input_data]
        zs = []

        for b, w in zip(self.biases, self.weights):
            # w * a + b
            z = np.dot(w, a) + b
            zs.append(z)
            # σ(z) = σ(w*a + b)
            a = sigmoid(z)
            activations.append(a)

        # (an - y) * σ'
        delta = (activations[-1] - expected_data) * sigmoid_derivative(zs[-1])

        # last nabla is computed
        nb_b[-1] = delta
        nb_w[-1] = np.dot(delta, activations[-2].T)

        # Compute all hidden layers
        for l in range(2, self.layer_count):
            z = zs[-l]
            sd = sigmoid_derivative(z)

            delta = np.dot(self.weights[-l + 1].T, delta) * sd
            nb_b[-l] = delta
            nb_w[-l] = np.dot(delta, activations[-l - 1].transpose())

        return nb_b, nb_w
