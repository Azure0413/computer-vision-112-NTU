import cv2
import numpy as np
import h5py


class Kanade_Motion():
    def __init__():
        pass

    def get_motions(self, ref: np.ndarray, target: np.ndarray, seg_ref_file: h5py.File, seg_target_file: h5py.File, grid=2):
        Hs = [np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)]
        Hs.extend(self._get_motion_row(ref, target, grid))
        Hs.extend(self._get_motion_col(ref, target, grid))
        Hs.extend(self._get_motion_block(ref, target, grid))
        Hs.extend(self._get_motion_seg(ref, target, seg_ref_file))

        return np.array(Hs)