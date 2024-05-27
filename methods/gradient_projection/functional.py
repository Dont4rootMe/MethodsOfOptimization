import numpy as np
import scipy.integrate


class FunctionalOracle:
    def _operator_A(self, x: np.array) -> np.array:
        """
        :param x: np.array vectorized function
        :return np.array
        """
        tmp = np.zeros(len(x))
        for t in range(0, len(x)):
            tmp[t] = scipy.integrate.simpson(x[:t+1], dx=1/len(x))

        return tmp

    def _conjugate_operator_A(self, x) -> np.array:
        """
        :param x: np.array vectorized function
        :return np.array
        """
        tmp = np.zeros(len(x))
        for t in range(0, len(x)):
            tmp[t] = scipy.integrate.simpson(x[t:], dx=1/len(x))

        return tmp

    def __init__(self, f: np.array, g: np.array, k: np.array):
        """
        param: f: np.array vectorized function
        param: grad_f: np.array vectorized gradient function
        """
        self.f = f
        self.g = g
        self.k = k
    
    def __call__(self, x: np.array) -> float:
        """
        param: x: np.array vectorized function
        return: float
        """
        residual = scipy.integrate.simpson((self._operator_A(x) - self.f) ** 2)
        # scalar_4 = scipy.integrate.simpson(self.g * x) ** 4
        scalar_2 = scipy.integrate.simpson(self._operator_A(x) * self.k)

        return residual + scalar_2

    def grad(self, x: np.array) -> np.array:
        """
        param: x: np.array vectorized function
        return: np.array
        """
        residual_grad = 2 * self._conjugate_operator_A(self._operator_A(x) - self.f)
        scalar_4_grad = 4 * self.g * scipy.integrate.simpson(self.g * x, dx=1/len(x)) ** 3
        scalar_2_grad = self._conjugate_operator_A(self.k)

        return residual_grad + scalar_2_grad