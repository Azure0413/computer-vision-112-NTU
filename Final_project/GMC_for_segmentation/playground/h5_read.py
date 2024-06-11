import h5py
import numpy as np
import cv2

SAM_path = "../DL_masks/SAM_masks_h5"
YOLO_path = "../DL_masks/YOLO_masks_h5"

with h5py.File(SAM_path+"/016.h5", 'r') as file:
    keys = list(file.keys())
    for key in keys:
        print(file[key])

    for i in range(file["areas"].shape[0]):
        print(file["areas"][i])
        print(np.sum(file["segmentations"][i]))
        arr = file["segmentations"][i]
        # Save as image
        cv2.imwrite(f"test{i}_{file['stability_scores'][i]}.png", arr*255)