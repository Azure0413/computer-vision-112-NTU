import numpy as np
from cvfgmc.utils import evaluate, gridding
import cv2
import scipy
import numba as nb

def homography_filter(Hs: np.ndarray):
    # Check if H is obvisouly wrong
    for i, H in enumerate(Hs):
        if H is None:
            continue
    Hs = np.array([H for H in Hs if H is not None])

    return Hs

def homography_optimization(ref: np.ndarray, H: np.ndarray, target: np.ndarray, mask: np.ndarray):
    objective = _homography_loss_func(ref, H, target, mask)
    num_of_pix = np.sum(mask)*256
    H = H.flatten()
    print(f"Initial MSE: {objective(H)/num_of_pix}", end=", ")
    result = scipy.optimize.minimize(objective, H, method="Nelder-Mead")
    print(f"Final MSE: {result.fun/num_of_pix}")
    H = result.x.reshape(3, 3)
    return H

def _homography_loss_func(ref: np.ndarray, H: np.ndarray, targ:np.ndarray, mask: np.ndarray):
    reference = ref
    homography = H
    target = targ
    mask = mask
    def loss_func(H):
        H = H.reshape(3, 3)
        target_warped = cv2.warpPerspective(reference, H, (reference.shape[1], reference.shape[0]), borderValue=0)
        loss = (target_warped - target)**2
        loss = gridding(loss, block_size=(16, 16))
        loss = loss * mask[:, :, None, None]
        return np.sum(loss)
    return loss_func