from cvfgmc.utils import target_frames, target_frames_slow, hierarchical_b_order
import cv2 
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

frames = target_frames('../data')
block_frames = frames.reshape(129, 135, 16, 240, 16).transpose(0, 1, 3, 2, 4)

zero_block_count = 0
for i in range(135):
    for j in range(240):
        if i == 92 and j == 146:
            print(block_frames[0, i, j])
print(zero_block_count)

def process(order):
    target, ref0, ref1 = order
    if ref0 is None:
        return
    same_target_0 = block_frames[ref0] == block_frames[target]
    same_target_1 = block_frames[target] == block_frames[ref1]
    same_block_0 = np.all(same_target_0, axis=(2, 3)).astype(np.int64)
    same_block_1 = np.all(same_target_1, axis=(2, 3)).astype(np.int64)
    same_block_count_0 = np.sum(same_block_0)
    same_block_count_1 = np.sum(same_block_1)

    same_cover_target_0 = np.repeat(np.repeat(same_block_0, 16, axis=0), 16, axis=1)
    same_cover_target_1 = np.repeat(np.repeat(same_block_1, 16, axis=0), 16, axis=1)

    same_edge_target_0 = cv2.Canny(same_cover_target_0*255, 100, 200)
    same_edge_target_1 = cv2.Canny(same_cover_target_1*255, 100, 200)

    # Thicken the edges
    kernel = np.ones((3, 3), dtype=int)
    same_edge_target_0 = cv2.dilate(same_edge_target_0, kernel, iterations=1)
    same_edge_target_1 = cv2.dilate(same_edge_target_1, kernel, iterations=1)

    # Convert the original image to 3 channels
    target_frame0 = cv2.cvtColor(frames[target], cv2.COLOR_GRAY2BGR)
    target_frame1 = cv2.cvtColor(frames[target], cv2.COLOR_GRAY2BGR)

    # Apply the edge mask to the original image
    target_frame0[same_edge_target_0 == 255] = [0, 255, 0]
    target_frame1[same_edge_target_1 == 255] = [255, 0, 0]

    cv2.imwrite(f'sameblock/{target}_0_{same_block_count_0}.png', target_frame0)
    cv2.imwrite(f'sameblock/{target}_1_{same_block_count_1}.png', target_frame1)

if __name__ == "__main__":
    pass
    #with ThreadPoolExecutor() as executor:
    #    images = list(executor.map(process, hierarchical_b_order(0, 128, 32)))
    
    #for _ in images:
    #    pass