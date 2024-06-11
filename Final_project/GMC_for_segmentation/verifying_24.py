import numpy as np
from cvfgmc.utils import target_frames, hierarchical_b_order, evaluate, merge_compensations
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import h5py
import numpy as np

model_idx = 2

# ----------------------------------- Data/Parameter -----------------------------------
frames = target_frames("data") # (129, 2160, 3840)
num_images = frames.shape[0]
process_order, skip_target = hierarchical_b_order(0, 128, 32) # (129, 3)

Hs_best = np.load(f"output/{model_idx}/Hs.npy")


compensated = np.empty((num_images, 2160, 3840))
psnr_all = []
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
    psnr_all.append([target, final_psnr])
    print(f"Frame {target} PSNR: {final_psnr}")
psnr_all = np.array(psnr_all)
psnr_all = psnr_all[np.argsort(psnr_all[:, 0])]
print(f"Average PSNR: {np.mean(psnr_all[:, 1])}")

# ----------------------------------- Visualization -----------------------------------
# bin plot
plt.figure(figsize=(8, 6))
plt.hist(psnr_all[:, 1], bins=10)
plt.xlabel('PSNR')
plt.ylabel('Frequency')
plt.title('PSNR Distribution')
plt.grid(True)
plt.savefig(f"output/{model_idx}/psnr_12_model.png")

# bin plot where target is odd
psnr_odd = psnr_all[psnr_all[:, 0] % 2 == 1]
plt.figure(figsize=(8, 6))
plt.hist(psnr_odd[:, 1], bins=10)
plt.xlabel('PSNR')
plt.ylabel('Frequency')
plt.title('PSNR Distribution (Odd)')
plt.grid(True)
plt.savefig(f"output/{model_idx}/psnr_12_model_odd.png")

# bin plot where target is even
psnr_even = psnr_all[psnr_all[:, 0] % 2 == 0]
plt.figure(figsize=(8, 6))
plt.hist(psnr_even[:, 1], bins=10)
plt.xlabel('PSNR')
plt.ylabel('Frequency')
plt.title('PSNR Distribution (Even)')
plt.grid(True)
plt.savefig(f"output/{model_idx}/psnr_24_model_even.png")

# Save psnr to csv
np.savetxt(f"output/{model_idx}/psnr_24_model.csv", psnr_all, delimiter=",", fmt='%.5f')