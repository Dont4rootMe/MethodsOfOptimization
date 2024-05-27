import numpy as np

class ADMM:
    @staticmethod
    def MC_update_X(Y, lambda_, r):
        T = Y - lambda_ / r
        U, S, Vh = np.linalg.svd(T, full_matrices=False)
        return U @ np.diag(S - 1/r) @ Vh

    @staticmethod
    def MC_update_Y(X, lambda_, r, Z, mask):

        canvas = X + lambda_ / r
        canvas[*mask] = Z[*mask]

        return canvas

    @staticmethod
    def MC_ADMM(Z, mask, tol, max_iters, r, verbose = False):
        X = np.zeros_like(Z)
        Y = np.zeros_like(Z)

        X[*mask] = Z[*mask].copy()
        Y[*mask] = Z[*mask].copy()
        lambda_ = np.zeros_like(Z)

        iterations = range(max_iters)
        last_score = +np.inf

        for _ in iterations:
            X = ADMM.MC_update_X(Y, lambda_, r).copy()
            Y = ADMM.MC_update_Y(X, lambda_, r, Z, mask).copy()
            lambda_ = lambda_ + r * (X - Y)

            score = np.linalg.svd(X, compute_uv=False).sum()
            if np.abs(last_score - score) < tol:
                print('break on tol')
                break
            else:
                last_score = score
                # print(score)

        return X