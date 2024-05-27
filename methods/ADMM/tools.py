import numpy as np
from methods.ADMM.oracle import ADMM

def make_mask(dim0, dim1, quant=0.1):
    x_inds = np.arange(dim0)
    y_inds = np.arange(dim1)

    len_data = dim0 * dim1
    inds = np.arange(len_data)
    np.random.shuffle(inds)
    inds_to_have = np.array(list(set(np.arange(len_data)) - set(inds[:int(quant * len_data)])))

    x_inds = inds_to_have // dim1
    y_inds = inds_to_have %  dim1

    return [x_inds, y_inds]

def cure_image(img, mask, tol, steps, r):
    channels = np.zeros_like(img)
    for i in range(img.shape[2]):
        channels[:, :, i] = ADMM.MC_ADMM(img[:, :, i], mask[i], tol, steps, r, False)
    return channels