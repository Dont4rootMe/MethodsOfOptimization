import numpy as np

class PoisOracle:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def fun(self, weights):
        """Compute objective, gradient, hessian"""
        scalar = self.X @ weights
        return np.sum(np.exp(scalar) - self.y * scalar)

    def grad(self, weights):
        """Compute gradient."""
        return (np.exp(weights @ self.X.T) - self.y).T @ self.X

    def hess(self, weights):
        """Compute hessian."""
        lambda_ = np.exp(weights @ self.X.T)
        return (lambda_[:, None] * self.X).T @ self.X

    def hessp(self, weights, vector):
        """Compute `hessian-times-vector` product. For scipy.optim.solve"""
        return (np.exp(weights @ self.X.T)[:, None] * self.X).T @ self.X @ vector


class Newton_CGMethod:
    def __init__(self, w_init, f, grad, hess, alpha, epsilon, max_iter):
        """
        :param Oracle: class for oracle
        :param X: np.array dataset for pois regression
        :param w_init: np.array initialization for weihgts
        :param f: callable target function
        :param grad: callable gradient of f
        :param hess: callable hessian of f
        :param alpha: learning rate
        :param epsilon: float
        :param max_iter: int
        """
        self.w = w_init.copy()
        self.f = f
        self.grad = grad
        self.hess = hess
        self.alpha = alpha
        self.epsilon = epsilon
        self.max_iter = max_iter

    def run(self):
        for i in range(self.max_iter):
            delta_w = np.linalg.solve(self.hess(self.w), self.grad(self.w))
            if self.epsilon is not None and np.linalg.norm(delta_w, 2) < self.epsilon:
                break
            self.w -= self.alpha * delta_w

        return self.w
    
    def get_value(self):
        return self.f(self.w)