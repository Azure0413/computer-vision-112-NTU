from .base import MotionModel
from cvfgmc.utils import gridding, degridding
import numpy as np
from numba import cuda

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

class MatchBlock(MotionModel):
    def __init__(self, method='forward', block_size=(16, 16)):
        # If the dimension of window_size is 1, it will be converted to (window_size, window_size)
        self.method = method
        self.block_size = block_size

    def compensate(self, target: np.ndarray, ref0: np.ndarray, ref1: np.ndarray) -> np.ndarray:
        if self.method == 'backward':
            reference = ref0
        else:
            reference = ref1

        ssd_out = np.zeros((target.shape[0], target.shape[1], 2), dtype=np.int16)
        target = gridding(target, self.block_size).astype(np.float32)
        reference = gridding(reference, self.block_size).astype(np.float32)
        
        # Transfer to device
        target_device = cuda.to_device(np.ascontiguousarray(target))
        reference_device = cuda.to_device(np.ascontiguousarray(reference))
        ssd_out_device = cuda.to_device(np.ascontiguousarray(ssd_out))

        threads_per_block = (4, 4)
        blocks_per_grid_x = (target.shape[-4] + threads_per_block[0] - 1) // threads_per_block[0]
        blocks_per_grid_y = (target.shape[-3] + threads_per_block[1] - 1) // threads_per_block[1]
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

        ssd_block_kernel[blocks_per_grid, threads_per_block](target_device, reference_device, ssd_out_device)
        cuda.synchronize()
        ssd_out = ssd_out_device.copy_to_host()

        compensated = np.zeros_like(target)
        for i in range(target.shape[0]):
            for j in range(target.shape[1]):
                x, y = ssd_out[i, j]
                compensated[i, j] = reference[x, y]
        
        compensated = degridding(compensated, self.block_size)
        return compensated

