from __future__ import division

import numpy as np
from numpy.lib.stride_tricks import as_strided
import scipy.sparse
import scipy.sparse.linalg
from PIL import Image
from time import time

def _rolling_block(A, block=(3, 3)):
    """Applies sliding window to given matrix."""
    shape = (A.shape[0] - block[0] + 1, A.shape[1] - block[1] + 1) + block
    strides = (A.strides[0], A.strides[1]) + A.strides
    return as_strided(A, shape=shape, strides=strides)


def compute_laplacian(img, eps=10**(-16), win_rad=1):
    """Computes Matting Laplacian for a given image.
    Args:
        img: numpy matrix with input image
        eps: regularization parameter controlling alpha smoothness
            from Eq. 12 of the original paper. Defaults to 1e-7.
        win_rad: radius of window used to build Matting Laplacian (i.e.
            radius of omega_k in Eq. 12).
    Returns: sparse matrix holding Matting Laplacian.
    """

    win_size = (win_rad * 2 + 1) ** 2
    h, w = img.shape
    # Number of window centre indices in h, w axes
    c_h, c_w = h - 2 * win_rad, w - 2 * win_rad
    win_diam = win_rad * 2 + 1

    indsM = np.arange(h * w).reshape((h, w), order='F')
    ravelImg = img.reshape(h * w, 1, order='F')
    win_inds = _rolling_block(indsM, block=(win_diam, win_diam))
    win_inds = win_inds.reshape(c_h, c_w, win_size)
    win_inds = win_inds.reshape(-1, win_size)
    winI = ravelImg[win_inds]

    win_mu = np.mean(winI, axis=1, keepdims=True)
    win_var = np.einsum('...ji,...jk ->...ik', winI, winI) / \
        win_size - np.einsum('...ji,...jk ->...ik', win_mu, win_mu)

    inv = np.linalg.inv(win_var + (eps/win_size))

    X = np.einsum('...ij,...jk->...ik', winI - win_mu, inv)
    vals = np.eye(win_size) - (1.0/win_size) * \
        (1 + np.einsum('...ij,...kj->...ik', X, winI - win_mu))

    nz_indsCol = np.tile(win_inds, win_size).ravel()
    nz_indsRow = np.repeat(win_inds, win_size).ravel()
    nz_indsVal = vals.ravel()
    t0 = time()
    L = scipy.sparse.coo_matrix(
        (nz_indsVal, (nz_indsRow, nz_indsCol)), shape=(h*w, h*w))
    t1 = time()
    print('Runtime of generating a sparse matrix via SicPy:',t1-t0,'second.')
    return L


if __name__ == "__main__":
    lena = np.asarray(Image.open('lena.bmp'))
    ff = compute_laplacian(lena)