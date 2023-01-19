import numpy as np


class Relu:
    def forward(self, x):
        self.x = x
        return np.maximum(self.x, 0)

    def backward(self, eta):
        eta[self.x <= 0] = 0
        return eta
