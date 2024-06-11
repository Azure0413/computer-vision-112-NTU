import os
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

# Part 1: Load necessary images
def load_selected_frames(folder, selected_frames):
    frames = {}
    for frame_idx in tqdm(selected_frames):
        image_path = os.path.join(folder, f'{frame_idx:03d}.png')
        image = Image.open(image_path).convert('L')
        image = np.array(image)
        frames[frame_idx] = image
    return frames

selected_frames = [0, 16, 32]
frames = load_selected_frames('./gt', selected_frames)
image_000 = frames[0]
image_016 = frames[16]
image_032 = frames[32]

# Part 2: Hierarchical-B processing order
processing_order = [(16, 0, 32)]

# Part 3: Segmentation
selection_map = np.zeros((135, 240))
model_map = np.zeros((135, 240))

# Part 4: Compensation
# model 1 - pixel estimation
# pixel_map_000 = np.zeros((2160, 3840))
# pixel_map_032 = np.zeros((2160, 3840))
# mask_000 = np.isin(image_000, image_016)
# mask_032 = np.isin(image_032, image_016)

# pixel_map_000[mask_000] = image_000[mask_000]
# pixel_map_032[mask_032] = image_032[mask_032]

# num_nonzero_000 = np.sum(image_000 > 0)
# num_nonzero_032 = np.sum(image_032 > 0)

# print(f"Number of pixels > 0 in image_000: {num_nonzero_000}")
# print(f"Number of pixels > 0 in image_032: {num_nonzero_032}")

# model 2 - block estimation
def get_blocks(image, block_size=16):
    blocks = []
    for i in range(0, image.shape[0], block_size):
        for j in range(0, image.shape[1], block_size):
            block = image[i:i+block_size, j:j+block_size]
            blocks.append((i // block_size, j // block_size, block))
    return blocks

blocks_000 = get_blocks(image_000)
blocks_032 = get_blocks(image_032)
blocks_016 = get_blocks(image_016)

unique_blocks_016 = {tuple(block.flatten()) for _, _, block in blocks_016}

for idx, blocks in enumerate([blocks_000, blocks_032]):
    for i, j, block in blocks:
        if tuple(block.flatten()) in unique_blocks_016:
            selection_map[i, j] = 1

print(np.sum(selection_map > 0))

output_image = Image.open('./gt/016.png').convert('RGB')
draw = ImageDraw.Draw(output_image)

block_size = 16
for i in range(selection_map.shape[0]):
    for j in range(selection_map.shape[1]):
        if selection_map[i, j] == 1:
            x0 = j * block_size
            y0 = i * block_size
            x1 = x0 + block_size
            y1 = y0 + block_size
            draw.rectangle([x0, y0, x1, y1], fill=None, outline='red')

output_folder = './output'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

output_file = os.path.join(output_folder, '016_select.png')
output_image.save(output_file)

print("保存成功！")
