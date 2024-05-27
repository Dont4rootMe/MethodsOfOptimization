import numpy as np


class SimplexMethod:
    def __init__(self, c: np.array, A: np.array, b: np.array, max_iter: int):
        """
        :param c: np.ndarray vector in <c, x> target linear function
        :param A: np.ndarray matrix of linear conditions Ax <= b
        :param b: np.ndarray vector of condition right side Ax <= b
        :param max_iter: int max count of iters
        """
        self.target_linear_func = c.copy()

        A_added =  np.concatenate([A.copy(), np.eye(len(b)), b.copy().reshape(-1, 1)], axis=-1)
        c_added = np.concatenate([-c.copy(), np.zeros((len(b) + 1, ))])
        self.matrix = np.concatenate([A_added, c_added.reshape(1, -1)], axis=0)
        self.basis = [i for i in range(A.shape[-1], self.matrix.shape[-1] - 1)]
        self.count_of_target_vars = A.shape[1]

        self.max_iter = max_iter

    def solve(self):
        matrix = self.matrix

        for _ in range(self.max_iter):
            if np.all(matrix[-1, :-1] >= 0):
                break

            leading_column = np.argmin(matrix[-1, :-1])

            convergence_speed = matrix[:-1, -1] / matrix[:-1, leading_column]
            convergence_speed[convergence_speed <= 0] = np.infty
            leading_row = np.argmin(convergence_speed)

            save_leading_row = (matrix[leading_row, :] / matrix[leading_row, leading_column]).copy()
            matrix[np.arange(matrix.shape[0]), :] -= matrix[leading_row, :][None, :] * (matrix[np.arange(matrix.shape[0]), leading_column] / matrix[leading_row, leading_column])[:, None]
            matrix[leading_row, :] = save_leading_row

            self.basis[leading_row] = leading_column

        self.matrix = matrix.copy()
        vec_optim = matrix[:-1, -1]
        answ = np.zeros((self.count_of_target_vars))
        for i, coord in enumerate(self.basis):
            if coord < self.count_of_target_vars:
                answ[coord] = vec_optim[i]

        return answ, self.matrix[-1, -1]
    
    def get_value(self):
        vec_optim = self.matrix[:, -1]
        answ = np.zeros((self.count_of_target_vars))
        for i, coord in enumerate(self.basis):
            if coord < self.count_of_target_vars:
                answ[coord] = vec_optim[i]

        return np.dot(self.target_linear_func, answ)

            

            




