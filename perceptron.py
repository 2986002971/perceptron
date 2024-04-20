from numba import jit
import numpy as np
import time


@jit(nopython=True)
def update_weights_numba(x, y, weights, bias, learning_rate):
    for idx, x_i in enumerate(x):
        output = np.dot(x_i, weights) + bias
        y_predicted = 1 if output > 0 else -1

        if y_predicted * y[idx] < 0:
            weights += learning_rate * y[idx] * x_i
            bias += learning_rate * y[idx]
    return weights, bias


class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=10):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, x, y):
        start_time = time.time()

        n_samples, n_features = x.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        y = np.array([1 if i > 0 else -1 for i in y])

        for _ in range(self.n_iters):
            self.weights, self.bias = update_weights_numba(x, y, self.weights, self.bias, self.learning_rate)

        print(f"Training time: {time.time() - start_time:.2f}s")

    def predict(self, x):
        out = np.dot(x, self.weights) + self.bias
        y_predict = np.where(out > 0, 1, -1)
        return y_predict


class DataSet:
    def __init__(self, dimension=2):
        self.dimension = dimension
        self.x = np.zeros((0, self.dimension))
        self.y = np.zeros(0)

    def gaussian_generate(self, center1, center2, sigma, num):
        if center1.shape[0] != self.dimension or center2.shape[0] != self.dimension:
            raise ValueError("Center dimension not match.")

        for _ in range(num):
            if np.random.random() < 0.5:
                x = center1 + np.random.randn(self.dimension) * sigma
                y = 1
            else:
                x = center2 + np.random.randn(self.dimension) * sigma
                y = -1

            self.x = np.vstack((self.x, x))
            self.y = np.hstack((self.y, y))


