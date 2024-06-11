import numpy as np
import cv2.ximgproc as xip
import cv2
import random
random.seed(999)

def computeDisp(Il, Ir, max_disp):
    h, w, ch = Il.shape
    labels = np.zeros((h, w), dtype=np.float32)
    Il = Il.astype(np.float32)
    Ir = Ir.astype(np.float32)

    # >>> Cost Computation
    # TODO: Compute matching cost
    # [Tips] Census cost = Local binary pattern -> Hamming distance
    # [Tips] Set costs of out-of-bound pixels = cost of closest valid pixel  
    # [Tips] Compute cost both Il to Ir and Ir to Il for later left-right consistency
    img_left = cv2.copyMakeBorder(Il,1,1,1,1, cv2.BORDER_CONSTANT, value=0)
    img_right = cv2.copyMakeBorder(Ir,1,1,1,1, cv2.BORDER_CONSTANT, value=0)
    
    img_left_bin = np.zeros((9, *img_left.shape))
    img_right_bin = np.zeros((9, *img_right.shape))
    idx = 0
    
    for x in range(-1, 2):
        for y in range(-1, 2):
            maskL = (img_left > np.roll(img_left, [y, x], axis=[0, 1]))
            img_left_bin[idx][maskL] = 1
            maskR = (img_right > np.roll(img_right, [y, x], axis=[0, 1]))
            img_right_bin[idx][maskR] = 1
            idx += 1
    
    img_left_bin = img_left_bin[:, 1:-1, 1:-1] 
    img_right_bin = img_right_bin[:, 1:-1, 1:-1]
    # >>> Cost Aggregation
    # TODO: Refine the cost according to nearby costs
    # [Tips] Joint bilateral filter (for the cost of each disparty)
    l_cost_volume = np.zeros((max_disp+1, h, w))
    r_cost_volume = np.zeros((max_disp+1, h, w))
    wndw_size = -1
    
    for d in range(max_disp+1):
        l_shift = img_left_bin[:, :, d:].astype(np.uint32)
        r_shift = img_right_bin[:, :, :w-d].astype(np.uint32)
        # Hamming distance
        cost = np.sum(l_shift^r_shift, axis=0)
        cost = np.sum(cost, axis=2).astype(np.float32) 
        l_cost = cv2.copyMakeBorder(cost, 0, 0, d, 0, cv2.BORDER_REPLICATE)
        l_cost_volume[d] = xip.jointBilateralFilter(Il, l_cost, wndw_size, 4, 9)
        r_cost = cv2.copyMakeBorder(cost, 0, 0, 0, d, cv2.BORDER_REPLICATE)
        r_cost_volume[d] = xip.jointBilateralFilter(Ir, r_cost, wndw_size, 4, 9)
    

    # >>> Disparity Optimization
    # TODO: Determine disparity based on estimated cost.
    # [Tips] Winner-take-all
    left_disp_map = np.argmin(l_cost_volume, axis=0)
    right_disp_map = np.argmin(r_cost_volume, axis=0)
    
    # >>> Disparity Refinement
    # TODO: Do whatever to enhance the disparity map
    # [Tips] Left-right consistency check -> Hole filling -> Weighted median filtering
    check = np.zeros((h, w), dtype=np.float32)
    x, y = np.meshgrid(range(w),range(h))
    r_x = (x - left_disp_map) 
    masked = (r_x >= 0)
    left_disp = left_disp_map[masked]
    right_disp = right_disp_map[y[masked], r_x[masked]]
    consist_maek = (left_disp == right_disp)
    check[y[masked][consist_maek], x[masked][consist_maek]] = left_disp_map[masked][consist_maek]
    
    check_pad = cv2.copyMakeBorder(check,0,0,1,1, cv2.BORDER_CONSTANT, value=max_disp)
    left_labels = np.zeros((h, w), dtype=np.float32)
    right_labels = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            idx_L, idx_R = 0, 0
            while check_pad[y, x+1-idx_L] == 0:
                idx_L += 1
            left_labels[y, x] = check_pad[y, x+1-idx_L]
            while check_pad[y, x+1+idx_R] == 0:
                idx_R += 1
            right_labels[y, x] = check_pad[y, x+1+idx_R]
    labels = np.min((left_labels, right_labels), axis=0)
    labels = xip.weightedMedianFilter(Il.astype(np.uint8), labels, 20)

    return labels.astype(np.uint8)