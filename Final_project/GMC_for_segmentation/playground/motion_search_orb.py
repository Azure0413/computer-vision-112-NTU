import numpy as np
from cvfgmc.utils import target_frames, hierarchical_b_order, evaluate, save_results, merge_compensations, motion_model_refining
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import h5py
import numpy as np

SAM_path = "../DL_masks/SAM_masks_h5"
YOLO_path = "../DL_masks/YOLO_masks_h5"

def search_motion(ref: np.ndarray, target: np.ndarray, seg_ref: h5py.Dataset, seg_target: h5py.Dataset):    
    orb = cv2.ORB_create(nfeatures=20000)
    keypoints_tar, descriptors_tar = orb.detectAndCompute(target, None)
    keypoints_ref, descriptors_ref = orb.detectAndCompute(ref, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)
    matches = bf.match(descriptors_ref, descriptors_tar)

    # Hs have a initial homography matrix that doesn't do any transformation
    Hs = [np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)]

    for i, seg in enumerate(seg_ref):
        clusters = []
        for m in matches:
            if seg[int(keypoints_ref[m.queryIdx].pt[1]), int(keypoints_ref[m.queryIdx].pt[0])]:
                clusters.append(m)
        points_ref = np.array([keypoints_ref[m.queryIdx].pt for m in clusters])
        points_tar = np.array([keypoints_tar[m.trainIdx].pt for m in clusters])
        if len(points_ref) < 4 or len(points_tar) < 4:
            continue
        H, _ = cv2.findHomography(points_ref, points_tar, cv2.RANSAC, ransacReprojThreshold=6.0, maxIters=1000, confidence=0.9999)

        if H is not None:
            Hs.append(H)

    return np.array(Hs)


# ----------------------------------- Data/Parameter -----------------------------------
frames = target_frames("../data") # (129, 2160, 3840)
num_images = frames.shape[0]
process_order, skip_target = hierarchical_b_order(0, 128, 32) # (129, 3)

final_compensated = np.empty((num_images, 2160, 3840), dtype=np.uint8)
Hs_best = np.empty((129, 2, 12, 3, 3), dtype=np.float64)
for target, ref0, ref1 in process_order:
    if ref0 is None or ref1 is None or target == 41:
        continue
    seg_target_file = h5py.File(SAM_path+f"/{target:03d}.h5", 'r')
    seg_target = seg_target_file["segmentations"]
    seg_ref0_file = h5py.File(SAM_path+f"/{ref0:03d}.h5", 'r')
    seg_ref0 = seg_ref0_file["segmentations"]
    seg_ref1_file = h5py.File(SAM_path+f"/{ref1:03d}.h5", 'r')
    seg_ref1 = seg_ref1_file["segmentations"]

    Hs_ref0 = search_motion(frames[ref0], frames[target], seg_ref0, seg_target)
    Hs_ref0_count = len(Hs_ref0)
    Hs_ref1 = search_motion(frames[ref1], frames[target], seg_ref1, seg_target)
    Hs_ref1_count = len(Hs_ref1)
    Hs_all = np.concatenate((Hs_ref0, Hs_ref1), axis=0)

    psnr = np.empty((Hs_ref0_count + Hs_ref1_count, ))
    masks = np.empty((Hs_ref0_count + Hs_ref1_count, 135, 240))
    block_mse = np.empty((Hs_ref0_count + Hs_ref1_count, 135, 240))
    compensated = np.empty((Hs_ref0_count + Hs_ref1_count, 2160, 3840))

    for i, H in enumerate(Hs_all):
        if i < Hs_ref0_count:
            compensated[i] = cv2.warpPerspective(frames[ref0], H, (frames[ref0].shape[1], frames[ref0].shape[0]))
        else:
            compensated[i] = cv2.warpPerspective(frames[ref1], H, (frames[ref1].shape[1], frames[ref1].shape[0]))

    for i, comp in enumerate(compensated):
        psnr[i], masks[i], block_mse[i] = evaluate(comp, frames[target], block_size=(16, 16), top=13000)
    
    Hs_ref0_idx, mse_sum_ref0 = motion_model_refining(block_mse[:len(Hs_ref0)], top_k_model=12, top_p_block=13000)
    Hs_ref1_idx, mse_sum_ref1 = motion_model_refining(block_mse[len(Hs_ref0):], top_k_model=12, top_p_block=13000)

    chosen_idx = np.concatenate((Hs_ref0_idx, Hs_ref0_count + Hs_ref1_idx), axis=0)
    psnr = psnr[chosen_idx]
    masks = masks[chosen_idx]
    block_mse = block_mse[chosen_idx]
    compensated = compensated[chosen_idx]

    Hs_best[target, 0] = Hs_ref0[Hs_ref0_idx]
    Hs_best[target, 1] = Hs_ref1[Hs_ref1_idx]
    np.save("Hs_all.npy", Hs_best)

    print(f"Total H: {len(Hs_ref0)+len(Hs_ref1)} Hs_ref0: {Hs_ref0_idx} Hs_ref1: {Hs_ref1_idx}")

    compensated = compensated.reshape(-1, 1, 2160, 3840)
    final_compensated[target] = merge_compensations(compensated, psnr, masks, block_mse)[0]
    final_psnr, final_mask, final_block_mse = evaluate(final_compensated[target], frames[target], block_size=(16, 16), top=13000)
    print(f"Frame {target} PSNR: {final_psnr}")
    cv2.imwrite(f"motion_search/{target:03d}.png", final_compensated[target])

    seg_target_file.close()
    seg_ref0_file.close()
    seg_ref1_file.close()

final_psnr, final_mask, final_block_mse = evaluate(final_compensated, frames, block_size=(16, 16), top=13000)

save = True
if save:
    save_results(process_order, final_compensated, final_mask, "solution_nomask", apply_mask=False)
    save_results(process_order, final_compensated, final_mask, "solution", apply_mask=True)
    print("Results saved")