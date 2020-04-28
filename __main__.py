from keras.datasets import mnist
from keras.utils import to_categorical

from NeuralNetwork import NeuralNetwork


def main():
    (x_train, y_train), (x_test, y_test) = prepare_data()

    neural_network = NeuralNetwork([784, 128, 10])

    neural_network.train_SGD(x_train, y_train, 30, 10, 0.4, test_data=[x_test, y_test])


def prepare_data():
    (x_train_orig, y_train_orig), (x_test_orig, y_test_orig) = mnist.load_data()

    y_tr_resh = y_train_orig.reshape(60000, 1)
    y_te_resh = y_test_orig.reshape(10000, 1)
    y_tr_T = to_categorical(y_tr_resh, num_classes=10)
    y_te_T = to_categorical(y_te_resh, num_classes=10)
    y_train = y_tr_T.T
    y_test = y_te_T.T

    x_train_flatten = x_train_orig.reshape(x_train_orig.shape[0], -1).T
    x_test_flatten = x_test_orig.reshape(x_test_orig.shape[0], -1).T
    x_train = x_train_flatten / 255.
    x_test = x_test_flatten / 255.

    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    main()
