import numpy as np
from cvfgmc.utils import target_frames, hierarchical_b_order, evaluate, merge_compensations, motion_model_refining, motion_model_refining_accurate, homography_optimization
import cv2

Hs_path = "../output/2/"

frames = target_frames("../data")
num_images = frames.shape[0]
process_order, skip_target = hierarchical_b_order(0, 128, 32)
Hs_best = np.load(Hs_path + "Hs.npy")

compensated = np.empty((num_images, 2160, 3840))
for target, ref0, ref1 in process_order:
    target_compensated = np.empty((24, 2160, 3840))
    psnr = np.empty((24))
    masks = np.empty((24, 135, 240))
    block_mse = np.empty((24, 135, 240))
    if ref0 is None or ref1 is None:
        continue

    for i, Hs in enumerate(Hs_best[target]):
        if i == 0:
            for j, H in enumerate(Hs):
                target_compensated[i*12+j] = cv2.warpPerspective(frames[ref0], H, (frames[ref0].shape[1], frames[ref0].shape[0]), borderValue=0)
                psnr[i*12+j], masks[i*12+j], block_mse[i*12+j] = evaluate(target_compensated[i*12+j], frames[target], block_size=(16, 16), top=13000)
        else:
            for j, H in enumerate(Hs):
                target_compensated[i*12+j] = cv2.warpPerspective(frames[ref1], H, (frames[ref1].shape[1], frames[ref1].shape[0]), borderValue=0)
                psnr[i*12+j], masks[i*12+j], block_mse[i*12+j] = evaluate(target_compensated[i*12+j], frames[target], block_size=(16, 16), top=13000)

    target_compensated = target_compensated.reshape(-1, 1, 2160, 3840)
    psnr = psnr.reshape(-1)
    masks = masks.reshape(-1, 135, 240)
    block_mse = block_mse.reshape(-1, 135, 240) 
    compensated[target], compensated_selection = merge_compensations(target_compensated, psnr, masks, block_mse)
    final_psnr, final_mask, final_block_mse = evaluate(compensated[target], frames[target], block_size=(16, 16), top=13000)
    print(f"Frame {target} Original PSNR: {final_psnr}")
    for i, Hs in enumerate(Hs_best[target]):
        if i == 0:
            for j, H in enumerate(Hs):
                Hs_best[target][i][j] = homography_optimization(frames[ref0], H, frames[target], compensated_selection == i*12+j)
        else:
            for j, H in enumerate(Hs):
                Hs_best[target][i][j] = homography_optimization(frames[ref1], H, frames[target], compensated_selection == i*12+j)

    target_compensated = np.empty((24, 2160, 3840))
    psnr = np.empty((24))
    masks = np.empty((24, 135, 240))
    block_mse = np.empty((24, 135, 240))
    # Evaluate the optimized homographies
    for i, Hs in enumerate(Hs_best[target]):
        if i == 0:
            for j, H in enumerate(Hs):
                target_compensated[i*12+j] = cv2.warpPerspective(frames[ref0], H, (frames[ref0].shape[1], frames[ref0].shape[0]), borderValue=0)
                psnr[i*12+j], masks[i*12+j], block_mse[i*12+j] = evaluate(target_compensated[i*12+j], frames[target], block_size=(16, 16), top=13000)
        else:
            for j, H in enumerate(Hs):
                target_compensated[i*12+j] = cv2.warpPerspective(frames[ref1], H, (frames[ref1].shape[1], frames[ref1].shape[0]), borderValue=0)
                psnr[i*12+j], masks[i*12+j], block_mse[i*12+j] = evaluate(target_compensated[i*12+j], frames[target], block_size=(16, 16), top=13000)

    target_compensated = target_compensated.reshape(-1, 1, 2160, 3840)
    psnr = psnr.reshape(-1)
    masks = masks.reshape(-1, 135, 240)
    block_mse = block_mse.reshape(-1, 135, 240) 
    compensated[target], compensated_selection = merge_compensations(target_compensated, psnr, masks, block_mse)
    final_psnr, final_mask, final_block_mse = evaluate(compensated[target], frames[target], block_size=(16, 16), top=13000)
    print(f"Frame {target} Final PSNR: {final_psnr}")

np.save(Hs_path + "Hs_optimized.npy", Hs_best)
