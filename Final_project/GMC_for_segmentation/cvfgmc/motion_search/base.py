import cv2
import numpy as np
import h5py


class FeatureModel():
    def __init__(self):
        pass

    def get_motions(self, ref: np.ndarray, target: np.ndarray, seg_ref_file: h5py.File, seg_target_file: h5py.File, grid=2):
        Hs = [np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)]
        kp_tar, des_tar, kp_ref, des_ref = self._get_all_keypoints(ref, target)
        Hs.extend(self._get_motion_row(kp_ref, des_ref, kp_tar, des_tar, ref, target, grid))
        Hs.extend(self._get_motion_col(kp_ref, des_ref, kp_tar, des_tar, ref, target, grid))
        Hs.extend(self._get_motion_block(kp_ref, des_ref, kp_tar, des_tar, ref, target, grid))
        Hs.extend(self._get_motion_seg(kp_ref, des_ref, kp_tar, des_tar, ref, target, seg_ref_file))

        return np.array(Hs)

    def _get_all_keypoints(self, ref: np.ndarray, target: np.ndarray):
        keypoints_tar, descriptors_tar = self.feature_detector.detectAndCompute(target, None)
        keypoints_ref, descriptors_ref = self.feature_detector.detectAndCompute(ref, None)
        return keypoints_tar, descriptors_tar, keypoints_ref, descriptors_ref

    def _get_motion_row(self, kp_ref, des_ref, kp_tar, des_tar, ref: np.ndarray, target: np.ndarray, grid=2):
        Hs = []
        for grid_size in range(1, grid+1):
            for h in range(grid_size):
                low_y = h*ref.shape[0]//grid
                high_y = (h+1)*ref.shape[0]//grid
                kp_ref_idx = [i for i, kp in enumerate(kp_ref) if kp.pt[1] >= low_y and kp.pt[1] < high_y]
                kp_ref_sub = [kp_ref[i] for i in kp_ref_idx]
                des_ref_sub = des_ref[kp_ref_idx]
                kp_tar_idx = [i for i, kp in enumerate(kp_tar) if kp.pt[1] >= low_y and kp.pt[1] < high_y]
                kp_tar_sub = [kp_tar[i] for i in kp_tar_idx]
                des_tar_sub = des_tar[kp_tar_idx]
                if len(kp_tar_sub) == 0 or len(kp_ref_sub) == 0:
                    continue

                sub_matches = self.BFMatcher.match(des_ref_sub, des_tar_sub)
                sub_matches = self._filter_matches(sub_matches)
                src_pts = np.float64([kp_ref_sub[m.queryIdx].pt for m in sub_matches]).reshape(-1, 1, 2)
                dst_pts = np.float64([kp_tar_sub[m.trainIdx].pt for m in sub_matches]).reshape(-1, 1, 2)
                if len(src_pts) < 4 or len(dst_pts) < 4:
                    continue
                H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=6.0, maxIters=1000, confidence=0.9999)
                if H is not None:
                    Hs.append(H)
        return Hs
    
    def _get_motion_col(self, kp_ref, des_ref, kp_tar, des_tar, ref: np.ndarray, target: np.ndarray, grid=2):
        Hs = []
        for grid_size in range(1, grid+1):

            for w in range(grid_size):
                low_x = w*ref.shape[1]//grid
                high_x = (w+1)*ref.shape[1]//grid

                kp_ref_idx = [i for i, kp in enumerate(kp_ref) if kp.pt[0] >= low_x and kp.pt[0] < high_x]
                kp_ref_sub = [kp_ref[i] for i in kp_ref_idx]
                des_ref_sub = des_ref[kp_ref_idx]
                kp_tar_idx = [i for i, kp in enumerate(kp_tar) if kp.pt[0] >= low_x and kp.pt[0] < high_x]
                kp_tar_sub = [kp_tar[i] for i in kp_tar_idx]
                des_tar_sub = des_tar[kp_tar_idx]
                if len(kp_tar_sub) == 0 or len(kp_ref_sub) == 0:
                    continue

                sub_matches = self.BFMatcher.match(des_ref_sub, des_tar_sub)
                sub_matches = self._filter_matches(sub_matches)
                src_pts = np.float64([kp_ref_sub[m.queryIdx].pt for m in sub_matches]).reshape(-1, 1, 2)
                dst_pts = np.float64([kp_tar_sub[m.trainIdx].pt for m in sub_matches]).reshape(-1, 1, 2)
                if len(src_pts) < 4 or len(dst_pts) < 4:
                    continue
                H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=6.0, maxIters=1000, confidence=0.9999)
                if H is not None:
                    Hs.append(H)
        return Hs
    
    def _get_motion_block(self, kp_ref, des_ref, kp_tar, des_tar, ref: np.ndarray, target: np.ndarray, grid=2):
        Hs = []
        for grid_size in range(1, grid+1):
            for w in range(grid_size):
                for h in range(grid_size):
                    low_x = w*ref.shape[1]//grid
                    high_x = (w+1)*ref.shape[1]//grid
                    low_y = h*ref.shape[0]//grid
                    high_y = (h+1)*ref.shape[0]//grid

                    kp_ref_idx = [i for i, kp in enumerate(kp_ref) if kp.pt[0] >= low_x and kp.pt[0] < high_x and kp.pt[1] >= low_y and kp.pt[1] < high_y]
                    kp_ref_sub = [kp_ref[i] for i in kp_ref_idx]
                    des_ref_sub = des_ref[kp_ref_idx]
                    kp_tar_idx = [i for i, kp in enumerate(kp_tar) if kp.pt[0] >= low_x and kp.pt[0] < high_x and kp.pt[1] >= low_y and kp.pt[1] < high_y]
                    kp_tar_sub = [kp_tar[i] for i in kp_tar_idx]
                    des_tar_sub = des_tar[kp_tar_idx]
                    if len(kp_tar_sub) == 0 or len(kp_ref_sub) == 0:
                        continue

                    sub_matches = self.BFMatcher.match(des_ref_sub, des_tar_sub)
                    sub_matches = self._filter_matches(sub_matches)
                    src_pts = np.float64([kp_ref_sub[m.queryIdx].pt for m in sub_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float64([kp_tar_sub[m.trainIdx].pt for m in sub_matches]).reshape(-1, 1, 2)
                    if len(src_pts) < 4 or len(dst_pts) < 4:
                        continue

                    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=6.0, maxIters=1000, confidence=0.9999)
                    if H is not None:
                        Hs.append(H)

        return np.array(Hs)

    def _get_motion_seg(self, kp_ref, des_ref, kp_tar, des_tar, ref: np.ndarray, target: np.ndarray, seg_ref_file: h5py.File):
        seg_ref = seg_ref_file["segmentations"][:]
        Hs = []
        for i, seg in enumerate(seg_ref):
            inside_kp_idx = []
            for j, kp in enumerate(kp_ref):
                if seg[int(kp.pt[1]), int(kp.pt[0])]:
                    inside_kp_idx.append(j)
            if len(inside_kp_idx) < 4:
                continue
            inside_kp_ref = [kp_ref[j] for j in inside_kp_idx]
            inside_des_ref = des_ref[inside_kp_idx]
            matches = self.BFMatcher.match(inside_des_ref, des_tar)
            src_pts = np.float64([inside_kp_ref[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float64([kp_tar[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            if len(src_pts) < 4 or len(dst_pts) < 4:
                continue
            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=6.0, maxIters=1000, confidence=0.9999)
            if H is not None:
                Hs.append(H)
        
        return Hs

    def _filter_matches(self, matches, method='Lowe'):
        good = []

        for match in matches:
            if match.distance < 10:
                good.append(match)
        return matches



