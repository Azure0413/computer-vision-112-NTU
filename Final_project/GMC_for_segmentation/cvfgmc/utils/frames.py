import os
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
import numba as nb

def load_image(path: str) -> np.ndarray:
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def target_frames(path) -> np.ndarray:
    '''
    return:
    -------
        frames: np.ndarray
            (129, 2160, 3840) size numpy array
    '''
    frames = np.empty((129, 2160, 3840), dtype=np.uint8)
    
    image_paths = [os.path.join(path, f'{i:03d}.png') for i in range(129)]
    
    with ThreadPoolExecutor() as executor:
        images = list(executor.map(load_image, image_paths))
    
    for i, image in enumerate(images):
        frames[i] = image
        
    return np.array(frames)

def gridding(frames: np.ndarray, block_size: tuple) -> np.ndarray:
    '''
    params:
    -------
        frames: np.ndarray
            (..., Height, Width) size numpy array
        block_size: int
            size of block
    return:
    -------
        block_frames: np.ndarray
            (..., Height, Width, block_size, block_size) size numpy array
    '''
    if frames.shape[-2] % block_size[0] != 0 or frames.shape[-1] % block_size[1] != 0:
        raise ValueError(f"Invalid block size. Height and Width must be divisible by block_size. Got: {frames.shape[-2]} % {block_size} and {frames.shape[-1]} % {block_size}")
    new_shape = frames.shape
    new_shape = new_shape[:-2] + (frames.shape[-2] // block_size[0], block_size[0], frames.shape[-1] // block_size[1], block_size[1])
    axes_order = list(range(len(new_shape) - 4)) + [-4, -2, -3, -1] 
    return frames.reshape(new_shape).transpose(axes_order)

def degridding(block_frames: np.ndarray, block_size: tuple) -> np.ndarray:
    '''
    params:
    -------
        block_frames: np.ndarray
            (..., Height, Width, block_size, block_size) size numpy array
        block_size: tuple
            height and width of block
    return:
    -------
        frames: np.ndarray
            (..., Height, Width) size numpy array
    '''
    if block_frames.shape[-2:] != block_size:
        raise ValueError(f"Invalid block size. Expected {block_size}, got {block_frames.shape[-2:]}")
    
    height, width = block_frames.shape[-4]*block_size[0], block_frames.shape[-3]*block_size[1]
    axes_order = list(range(len(block_frames.shape) - 4)) + [-4, -2, -3, -1]
    return block_frames.transpose(axes_order).reshape(block_frames.shape[:-4] + (height, width))

def merge_compensations(compensateds: np.ndarray, psrns, masks, block_mses) -> np.ndarray:
    '''
    params:
    -------
        compensateds: np.ndarray
            (N, 129, 2160, 3840) size numpy array
        psrns: np.ndarray
            (N, 129) size numpy array
        masks: np.ndarray
            (N, 129, 135, 240) size numpy array
        block_mses: np.ndarray
            (N, 129, 135, 240) size numpy array
    '''
    arg_min_mses = np.argmin(block_mses, axis=0)
    compensateds = gridding(compensateds, (16, 16))
    img_idx, height_idx, width_idx = np.meshgrid(np.arange(compensateds.shape[1]), np.arange(compensateds.shape[2]), np.arange(compensateds.shape[3]), indexing='ij')
    compensateds = compensateds[arg_min_mses, img_idx, height_idx, width_idx]
    compensateds = degridding(compensateds, (16, 16))
    return compensateds, arg_min_mses

def mask_frame(target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    '''
    params:
    -------
        target: np.ndarray
            (..., Height, Width) size numpy array
        mask: np.ndarray
            (..., Height, Width) size numpy array
    return:
    -------
        masked: np.ndarray
            (..., Height, Width) size numpy array
    '''
    target = gridding(target, (16, 16))
    masked = np.where(mask[..., np.newaxis, np.newaxis], target, 0)
    masked = degridding(masked, (16, 16))
    return masked

def _save_one_result(target: int, target_block: np.ndarray, target_model_mask: np.ndarray, mask: np.ndarray, path: str):
    cv2.imwrite(f"{path}/{target:03d}.png", target_block[target])
    np.savetxt(f"{path}/s_{target:03d}.txt", mask[target].flatten(), delimiter='\n', fmt='%d')
    np.savetxt(f"{path}/m_{target:03d}.txt", target_model_mask[target].flatten(), delimiter='\n', fmt='%d')

def save_results(process_order: list, target_block: np.ndarray, model_map: np.ndarray, mask: np.ndarray, path: str, remove_folder=True, apply_mask=True):
    '''
    params:
    -------
        target_block: np.ndarray
            (..., Height, Width, block_height, block_width) size numpy array
        mask: np.ndarray
            (..., Height, Width, block_height, block_width) size numpy array
        path: str
            path to save the result
    '''
    print(f"### Saving results to {path}")

    if apply_mask:
        target_block = mask_frame(target_block, mask)

    if not os.path.exists(path):
        os.makedirs(path)

    with ThreadPoolExecutor() as executor:
        for target in range(len(process_order)):
            executor.submit(_save_one_result, target, target_block, model_map, mask, path)
    
    # Zip the solution folder in silent mode
    print("### Zipping the solution folder")
    os.system(f"zip -r -q {path}.zip {path}")

    # Remove the solution folder
    if remove_folder:
        print("### Removing the solution folder")
        os.system(f"rm -r {path}")


if __name__ == "__main__":
    fast = target_frames("../../data")
    assert np.all(fast == slow)

    block_frames = gridding(fast, (16, 16))
    deblock_frames = degridding(block_frames, (16, 16))
    assert np.all(fast == deblock_frames)