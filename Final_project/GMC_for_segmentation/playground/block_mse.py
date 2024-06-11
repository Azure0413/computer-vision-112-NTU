import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time
from cvfgmc.utils import gridding

target = np.random.randint(0, 256, (129, 2160, 3840), dtype=np.uint8)
target = gridding(target, (16, 16))
truth = np.random.randint(0, 256, (129, 2160, 3840), dtype=np.uint8)
truth = gridding(truth, (16, 16))
block_size = (16, 16)

srt = time.time()
mse_block = np.mean((target.astype(np.int32) - truth.astype(np.int32))**2, axis=(-2, -1))
print(time.time()-srt)

def mse_last_2d(target: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.mean((target.astype(np.int32) - truth.astype(np.int32))**2, axis=(-2, -1))

mse_block_separate = np.empty((target.shape[0:-2]), dtype=np.float64)

srt = time.time()
# Using ThreadPoolExecutor
def mse_last_2d_thread(target: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.mean((target.astype(np.int32) - truth.astype(np.int32))**2, axis=(-2, -1))

with ThreadPoolExecutor() as executor:
    mse_block_thread = list(executor.map(mse_last_2d_thread, target, truth))

for i, mse in enumerate(mse_block_thread):
    mse_block_separate[i] = mse

print(time.time()-srt)

print(np.all(mse_block == mse_block_separate))
