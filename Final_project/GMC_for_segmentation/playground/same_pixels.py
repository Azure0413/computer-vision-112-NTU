from cvfgmc.utils import target_frames, target_frames_slow, hierarchical_b_order
import cv2 
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

frames = target_frames('../data')
pixel_num = frames.shape[1] * frames.shape[2]
def process(order):
    target, ref0, ref1 = order
    if ref0 is None:
        return
    same_target_0 = frames[ref0] == frames[target]
    same_target_1 = frames[target] == frames[ref1]
    same_target_portion_0 = str(np.sum(same_target_0)/pixel_num).split('.')[1][:4]
    same_target_portion_1 = str(np.sum(same_target_1)/pixel_num).split('.')[1][:4]

    # Convert the original image to 3 channels
    target_frame0 = cv2.cvtColor(frames[target], cv2.COLOR_GRAY2BGR)
    target_frame1 = cv2.cvtColor(frames[target], cv2.COLOR_GRAY2BGR)

    # Apply the edge mask to the original image
    target_frame0[same_target_0 == True] = [0, 255, 0]
    target_frame1[same_target_1 == True] = [255, 0, 0]

    cv2.imwrite(f'samepixel/{target}_0_{same_target_portion_0}.png', target_frame0)
    cv2.imwrite(f'samepixel/{target}_1_{same_target_portion_1}.png', target_frame1)

if __name__ == "__main__":
    with ThreadPoolExecutor() as executor:
        images = list(executor.map(process, hierarchical_b_order(0, 128, 32)))
    
    for _ in images:
        pass