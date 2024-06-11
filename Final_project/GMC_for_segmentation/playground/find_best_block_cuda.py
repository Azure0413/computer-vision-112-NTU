import numpy as np
from cvfgmc.utils import gridding, degridding, evaluate
import time
import numba as nb
import cupy as cp
import cv2

np.random.seed(0)
truth = np.random.randint(0, 255, (2160, 3840), dtype=np.int32)
target = truth.copy()
reference = np.random.randint(0, 255, (2160, 3840), dtype=np.int32)

psnr, mask, block_mse = evaluate(truth, reference, (16, 16), 13000)
print(psnr)

target = gridding(target, (16, 16))
reference = gridding(reference, (16, 16))

target = cp.array(target, dtype=cp.float32)
reference = cp.array(reference, dtype=cp.float32)

def ssd_block(block: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return cp.sum(cp.square(reference - block), axis=(-2, -1))

total_srt = time.time()
num_blocks = 24
for i in range(0, target.shape[0]):
    srt = time.time()
    for j in range(0, target.shape[1], num_blocks):
        block = target[i, j:j+num_blocks]
        block = cp.expand_dims(block, axis=(1, 2))
        mse = ssd_block(block, reference)
        min_idx = cp.unravel_index(cp.argmin(mse, axis=(-1, -2)), mse.shape[1:])
        target[i, j:j+num_blocks] = reference[min_idx[0], min_idx[1]]

    print(f"Row {i} took {time.time() - srt} seconds")
print(f"Total time: {time.time() - total_srt} seconds")


target = cp.asnumpy(target)
reference = cp.asnumpy(reference)
print(target.shape)
print(reference.shape)

# Target is not (1, 2160, 3840). Make it (2160, 3840)

psnr, mask, block_mse = evaluate(truth, degridding(target, (16, 16)), (16, 16), 13000)
print(psnr)