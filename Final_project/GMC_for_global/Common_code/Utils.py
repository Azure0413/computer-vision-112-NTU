def Hierarchical_B():
    return [ (16, 0, 32), (8, 0, 16), (4, 0, 8), (2, 0, 4), (1, 0, 2),
    (3, 2, 4), (6, 4, 8), (5, 4, 6), (7, 6, 8), (12, 8, 16),
    (10, 8, 12), (9, 8, 10), (11, 10, 12), (14, 12, 16), 
    (13, 12, 14), (15, 14, 16), (24, 16, 32),
    (20, 16, 24), (18, 16, 20), (17, 16, 18), (19, 18, 20),
    (22, 20, 24), (21, 20, 22), (23, 22, 24), (28, 24, 32),
    (26, 24, 28), (25, 24, 26), (27, 26, 28), (30, 28, 32),
    (29, 28, 30), (31, 30, 32),
    (48, 32, 64), (40, 32, 48), (36, 32, 40), (34, 32, 36),
    (33, 32, 34), (35, 34, 36), (38, 36, 40), (37, 36, 38),
    (39, 38, 40), (44, 40, 48), (42, 40, 44), (41, 40, 42),
    (43, 42, 44), (46, 44, 48), (45, 44, 46), (47, 46, 48),
    (56, 48, 64), (52, 48, 56), (50, 48, 52),
    (49, 48, 50), (51, 50, 52), (54, 52, 56), (53, 52, 54),
    (55, 54, 56), (60, 56, 64), (58, 56, 60), (57, 56, 58),
    (59, 58, 60), (62, 60, 64), (61, 60, 62), (63, 62, 64),
    (80, 64, 96), (72, 64, 80), (68, 64, 72), (66, 64, 68),
    (65, 64, 66), (67, 66, 68), (70, 68, 72), (69, 68, 70),
    (71, 70, 72), (76, 72, 80), (74, 72, 76), (73, 72, 74),
    (75, 74, 76), (78, 76, 80), (77, 76, 78), (79, 78, 80),
    (88, 80, 96), (84, 80, 88), (82, 80, 84),
    (81, 80, 82), (83, 82, 84), (86, 84, 88), (85, 84, 86),
    (87, 86, 88), (92, 88, 96), (90, 88, 92), (89, 88, 90),
    (91, 90, 92), (94, 92, 96), (93, 92, 94), (95, 94, 96),
    (112, 96, 128), (104, 96, 112), (100, 96, 104), (98, 96, 100),
    (97, 96, 98), (99, 98, 100), (102, 100, 104), (101, 100, 102),
    (103, 102, 104), (108, 104, 112), (106, 104, 108), (105, 104, 106),
    (107, 106, 108), (110, 108, 112), (109, 108, 110), (111, 110, 112),
    (120, 112, 128), (116, 112, 120), (114, 112, 116),
    (113, 112, 114), (115, 114, 116), (118, 116, 120), (117, 116, 118),
    (119, 118, 120), (124, 120, 128), (122, 120, 124), (121, 120, 122),
    (123, 122, 124), (126, 124, 128), (125, 124, 126), (127, 126, 128)
]
    
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def gridding(frames: np.ndarray, block_size: tuple) -> np.ndarray:
    if frames.shape[-2] % block_size[0] != 0 or frames.shape[-1] % block_size[1] != 0:
        raise ValueError(f"Invalid block size. Height and Width must be divisible by block_size. Got: {frames.shape[-2]} % {block_size} and {frames.shape[-1]} % {block_size}")
    new_shape = frames.shape
    new_shape = new_shape[:-2] + (frames.shape[-2] // block_size[0], block_size[0], frames.shape[-1] // block_size[1], block_size[1])
    axes_order = list(range(len(new_shape) - 4)) + [-4, -2, -3, -1] 
    return frames.reshape(new_shape).transpose(axes_order)

def calculate_psnr(mse):
    if mse == 0:
        return 1000000
    return 10 * np.log10(255**2 / mse)

def _mse_last_2d(blocks):
    target, truth = blocks
    return np.mean((target.astype(np.int32) - truth.astype(np.int32)) ** 2, axis=(-2, -1))

def process_images(output_folder, gt_folder, txt_folder, num_pixels=13000, block_size=16):
    os.makedirs(txt_folder, exist_ok=True)

    for frame_idx in tqdm(range(129)):
        output_path = os.path.join(output_folder, f'{frame_idx:03d}.png')
        ground_truth_path = os.path.join(gt_folder, f'{frame_idx:03d}.png')

        if not os.path.exists(output_path) or not os.path.exists(ground_truth_path):
            continue

        output_image = Image.open(output_path).convert('L')
        ground_truth_image = Image.open(ground_truth_path).convert('L')

        output_array = np.array(output_image)
        ground_truth_array = np.array(ground_truth_image)

        height, width = output_array.shape
        num_blocks_height = height // block_size
        num_blocks_width = width // block_size

        # Reshape into blocks
        output_blocks = gridding(output_array, (block_size, block_size))
        ground_truth_blocks = gridding(ground_truth_array, (block_size, block_size))

        mse_block_grid = np.empty(output_blocks.shape[:-2], dtype=np.float64)

        # Calculate MSE for each block in parallel
        with ThreadPoolExecutor() as executor:
            blocked_mses = list(executor.map(_mse_last_2d, zip(output_blocks, ground_truth_blocks)))

        for i, mse in enumerate(blocked_mses):
            mse_block_grid[i] = mse

        mse_block_flat = mse_block_grid.reshape(-1, mse_block_grid.shape[-2] * mse_block_grid.shape[-1])
        mse_block_top_k_idx = np.argpartition(mse_block_flat, num_pixels, axis=-1)[:, :num_pixels]

        # Construct a bool mask for the top k blocks
        mse_block_mask = np.zeros_like(mse_block_flat, dtype=bool)
        mse_block_mask[np.arange(mse_block_mask.shape[0])[:, None], mse_block_top_k_idx] = True
        mse_block_mask = mse_block_mask.reshape(mse_block_grid.shape)

        mse_selected = np.where(mse_block_mask, mse_block_grid, 0)
        mse = np.sum(mse_selected, axis=(-2, -1)) / num_pixels
        psnr = 10 * np.log10(255**2 / mse)

        # Create selection map
        selection_map = mse_block_mask.astype(np.uint8)

        txt_path = os.path.join(txt_folder, f's_{frame_idx:03d}.txt')
        selection_map_flat = selection_map.flatten().astype(int)
        np.savetxt(txt_path, selection_map_flat, fmt='%d')

def Segmentation():
    output_folder = '../output'
    gt_folder = '../data'
    txt_folder = '../output'

    process_images(output_folder, gt_folder, txt_folder)

def load_selected_frames(folder):
    frames = {}
    selected_frames = [i for i in range(129)]
    for frame_idx in tqdm(selected_frames):
        image_path = os.path.join(folder, f'{frame_idx:03d}.png')
        image = Image.open(image_path).convert('L')
        image = np.array(image)
        frames[frame_idx] = image
    return frames

def save_frames_and_txt(frames, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for frame_idx, frame in tqdm(frames.items()):
        frame_path = os.path.join(output_folder, f'{frame_idx:03d}.png')
        Image.fromarray(frame).save(frame_path)
        
def divide_into_blocks(frame, block_size=16):
    height, width = frame.shape
    blocks = []
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            blocks.append((i, j, frame[i:i+block_size, j:j+block_size]))
    return blocks

def save_model_map(model_maps, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for frame_idx, model_map in model_maps.items():
        output_path = os.path.join(output_folder, f'm_{frame_idx:03d}.txt')
        with open(output_path, 'w') as f:
            for model_id in model_map:
                f.write(f"{model_id}\n")