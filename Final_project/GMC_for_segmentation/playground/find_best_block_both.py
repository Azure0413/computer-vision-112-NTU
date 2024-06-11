import numpy as np
from cvfgmc.utils import gridding, degridding, evaluate
import time
import numba as nb
from numba import cuda

np.random.seed(0)
truth = np.random.randint(0, 255, (2160, 3840), dtype=np.int32)
target = truth.copy()
reference = np.random.randint(0, 255, (2160, 3840), dtype=np.int32)

psnr, mask, block_mse = evaluate(truth, reference, (16, 16), 13000)
print(psnr)

target = gridding(target, (16, 16)).astype(np.float32)
reference = gridding(reference, (16, 16)).astype(np.float32)

@cuda.jit
def ssd_block_kernel(target, reference, ssd_idx):
    i, j = cuda.grid(2)
    if i < ssd_idx.shape[0] and j < ssd_idx.shape[1]:
        block = target[i, j]
        min_ssd = 16646400
        min_idx_x, min_idx_y = 0, 0
        for k in range(reference.shape[0]):
            for l in range(reference.shape[1]):
                ssd = 0.0
                for m in range(block.shape[0]):
                    for n in range(block.shape[1]):
                        ssd += (block[m, n] - reference[k, l, m, n])**2
                if ssd < min_ssd:
                    min_ssd = ssd
                    min_idx_x, min_idx_y = k, l
        ssd_idx[i, j] = min_idx_x, min_idx_y

ssd_out = np.zeros((target.shape[0], target.shape[1], 2), dtype=np.int16)

# Transfer to device
target_device = cuda.to_device(np.ascontiguousarray(target))
reference_device = cuda.to_device(np.ascontiguousarray(reference))
ssd_out_device = cuda.to_device(np.ascontiguousarray(ssd_out))

# Configure the blocks
threads_per_block = (16, 16)
blocks_per_grid_x = (target.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
blocks_per_grid_y = (target.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

# Launch the kernel
start_time = time.time()

ssd_block_kernel[blocks_per_grid, threads_per_block](target_device, reference_device, ssd_out_device)
cuda.synchronize()

print("CUDA kernel time:", time.time() - start_time)

ssd_out = ssd_out_device.copy_to_host()

# Replace blocks in target with the most similar blocks from reference
for i in range(target.shape[0]):
    for j in range(target.shape[1]):
        min_idx_x, min_idx_y = ssd_out[i, j]
        target[i, j] = reference[min_idx_x, min_idx_y]


psnr, mask, block_mse = evaluate(truth, degridding(target, (16, 16)), (16, 16), 13000)
print(psnr)