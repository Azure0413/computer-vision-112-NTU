import numpy as np
from cvfgmc.utils import gridding, degridding, evaluate
import time
import numba as nb

np.random.seed(0)
truth = np.random.randint(0, 255, (2160, 3840), dtype=np.int32)
target = truth.copy()
reference = np.random.randint(0, 255, (2160, 3840), dtype=np.int32)

psnr, mask, block_mse = evaluate(truth, reference, (16, 16), 13000)
print(psnr)

target = gridding(target, (16, 16))
reference = gridding(reference, (16, 16))

nb.set_num_threads(nb.config.NUMBA_NUM_THREADS - 1)
@nb.njit('int32[:, :] (int32[:, :], int32[:, :, :, :])', parallel=True)
def ssd_block(block: np.ndarray, reference: np.ndarray) -> np.ndarray:
    ssd = np.zeros((reference.shape[0], reference.shape[1]), dtype=np.int32)
    r_w, r_h = reference.shape[0], reference.shape[1]
    b_w, b_h = block.shape[0], block.shape[1]
    for i in nb.prange(r_w):
        for j in range(r_h):
            for k in range(b_w):
                for l in range(b_h):
                    ssd[i, j] += (block[k, l] - reference[i, j, k, l])**2
    return ssd

for i in range(0, target.shape[0]):
    srt = time.time()
    for j in range(0, target.shape[1]):
        block = target[i, j]
        #block_srt = time.time()
        mse = ssd_block(block, reference)
        #print(f"Block {i}, {j} took {time.time() - block_srt} seconds")
        min_idx = np.unravel_index(np.argmin(mse), mse.shape)
        target[i, j] = reference[min_idx]
    print(f"Row {i} took {time.time() - srt} seconds")

psnr, mask, block_mse = evaluate(truth, degridding(target, (16, 16)), (16, 16), 13000)
print(psnr)