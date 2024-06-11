import numpy as np
from cvfgmc.utils import target_frames, hierarchical_b_order, evaluate, save_results, merge_compensations, motion_model_refining, homography_filter
from cvfgmc.motion_search import ORB_Motion, AKAZE_Motion, SIFT_Motion, KAZE_Motion, BRISK_Motion
import cv2
import h5py
import numpy as np
import os
import time 

# ----------------------------------- Data/Parameter -----------------------------------
SAM_path = "DL_masks/SAM_masks_h5"
YOLO_path = "DL_masks/Yolo_masks_h5_track"
SEG_path = SAM_path
frames = target_frames("data") # (129, 2160, 3840)
num_images = frames.shape[0]
process_order, skip_target = hierarchical_b_order(0, 128, 32) # (129, 3)

otuput_time = time.strftime("%Y%m%d-%H%M%S")
output_dir = "output"
output_path = f"{output_dir}/{otuput_time}"
if not os.path.exists(output_path):
    os.makedirs(output_path)

# ----------------------------------- Motion Search -----------------------------------
motion_models = [ORB_Motion(), AKAZE_Motion(), SIFT_Motion(), BRISK_Motion()]

final_compensated = np.empty((num_images, 2160, 3840), dtype=np.uint8)
Hs_best = np.empty((129, 2, 12, 3, 3), dtype=np.float64)
for target, ref0, ref1 in process_order:
    if ref0 is None or ref1 is None:
        continue
    seg_target_file = h5py.File(SEG_path+f"/{target:03d}.h5", 'r')
    seg_ref0_file = h5py.File(SEG_path+f"/{ref0:03d}.h5", 'r')
    seg_ref1_file = h5py.File(SEG_path+f"/{ref1:03d}.h5", 'r')

    Hs_ref0 = []
    Hs_ref1 = []
    for model in motion_models:
        srt = time.time()
        Hs_ref0.append(model.get_motions(frames[ref0], frames[target], seg_ref0_file, seg_target_file, grid=5))
        Hs_ref1.append(model.get_motions(frames[ref1], frames[target], seg_ref1_file, seg_target_file, grid=5))
        spent = time.time() - srt
        print(f"{model.__class__.__name__}: {spent:.2f}s", end=", ")
    print()
    Hs_ref0 = np.concatenate(Hs_ref0, axis=0)
    Hs_ref1 = np.concatenate(Hs_ref1, axis=0)

    Hs_ref0_count = len(Hs_ref0)
    Hs_ref1_count = len(Hs_ref1)
    Hs_all = np.concatenate((Hs_ref0, Hs_ref1), axis=0)
    print(f"Total H: {len(Hs_ref0)+len(Hs_ref1)} Hs_ref0: {Hs_ref0_count} Hs_ref1: {Hs_ref1_count}")

    psnr = np.empty((Hs_ref0_count + Hs_ref1_count, ))
    masks = np.empty((Hs_ref0_count + Hs_ref1_count, 135, 240))
    block_mse = np.empty((Hs_ref0_count + Hs_ref1_count, 135, 240))

    for i, H in enumerate(Hs_all):
        if i < Hs_ref0_count:
            compensated = cv2.warpPerspective(frames[ref0], H, (frames[ref0].shape[1], frames[ref0].shape[0]))
        else:
            compensated = cv2.warpPerspective(frames[ref1], H, (frames[ref1].shape[1], frames[ref1].shape[0]))

        psnr[i], masks[i], block_mse[i] = evaluate(compensated, frames[target], block_size=(16, 16), top=13000)

    Hs_ref0_idx, mse_sum_ref0 = motion_model_refining(block_mse[:len(Hs_ref0)], top_k_model=12, top_p_block=13000)
    Hs_ref1_idx, mse_sum_ref1 = motion_model_refining(block_mse[len(Hs_ref0):], top_k_model=12, top_p_block=13000)

    chosen_idx = np.concatenate((Hs_ref0_idx, Hs_ref0_count + Hs_ref1_idx), axis=0)
    psnr = psnr[chosen_idx]
    masks = masks[chosen_idx]
    block_mse = block_mse[chosen_idx]
    compensated = np.empty((len(chosen_idx), 2160, 3840))
    for i, chosen_idx in enumerate(chosen_idx):
        if chosen_idx < Hs_ref0_count:
            compensated[i] = cv2.warpPerspective(frames[ref0], Hs_all[chosen_idx], (frames[ref0].shape[1], frames[ref0].shape[0]))
        else:
            compensated[i] = cv2.warpPerspective(frames[ref1], Hs_all[chosen_idx], (frames[ref1].shape[1], frames[ref1].shape[0]))

    Hs_best[target, 0] = Hs_ref0[Hs_ref0_idx]
    Hs_best[target, 1] = Hs_ref1[Hs_ref1_idx]
    np.save(f"{output_path}/Hs.npy", Hs_best)

    compensated = compensated.reshape(-1, 1, 2160, 3840)
    final_compensated[target] = merge_compensations(compensated, psnr, masks, block_mse)[0]
    final_psnr, final_mask, final_block_mse = evaluate(final_compensated[target], frames[target], block_size=(16, 16), top=13000)

    seg_target_file.close()
    seg_ref0_file.close()
    seg_ref1_file.close()

    print(f"Frame {target} PSNR: {final_psnr}")

final_psnr, final_mask, final_block_mse = evaluate(final_compensated, frames, block_size=(16, 16), top=13000)

save = True
if save:
    save_results(process_order, final_compensated, final_mask, f"{output_path}/solution_nomask", apply_mask=False)
    save_results(process_order, final_compensated, final_mask, f"{output_path}/solution", apply_mask=True)
    print("Results saved")