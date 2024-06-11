import numpy as np
from .base import MotionModel
import cv2 

class Perspective(MotionModel):
    def __init__(self, method: str = 'forward', histogram_matching: bool = False):
        
        if method not in ['forward', 'backward', 'exact']:
            raise ValueError(f"Invalid method. Expected 'forward' or 'backward', got {method}")
        
        self.method = method
        self.histogram_matching = histogram_matching
        

    def compensate(self, target: np.ndarray, ref0: np.ndarray, ref1: np.ndarray) -> np.ndarray:
        '''
        params:
        -------
            target: np.ndarray
                (Height, Width) size numpy array
            ref0: np.ndarray
                (Height, Width) size numpy array
            ref1: np.ndarray
                (Height, Width) size numpy array
        return:
        -------
            compensated: np.ndarray
                (Height, Width) size numpy array
        '''
        
        # Detect ORB keypoints and descriptors in all images
                # Detect ORB keypoints and descriptors in all images
        orb = cv2.ORB_create()
        
        if self.method == 'forward':
            ref = ref0
        elif self.method == 'backward':
            ref = ref1
        
        
        keypoints_target, descriptors_target = orb.detectAndCompute(target, None)
        keypoints_ref, descriptors_ref = orb.detectAndCompute(ref, None)
        
        # Use BFMatcher to find the best matches
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches0 = bf.match(descriptors_target, descriptors_ref)
        
        # Sort the matches based on distance
        matches0 = sorted(matches0, key=lambda x: x.distance)
    
        
        # Extract location of good matches
        points_target0 = np.zeros((len(matches0), 2), dtype=np.float32)
        points_ref = np.zeros((len(matches0), 2), dtype=np.float32)
        
        
        for i, match in enumerate(matches0):
            points_target0[i, :] = keypoints_target[match.queryIdx].pt
            points_ref[i, :] = keypoints_ref[match.trainIdx].pt
                
        # Find homography
        H0, _ = cv2.findHomography(points_ref, points_target0, cv2.RANSAC)
        
        height, width = target.shape
        
        # Warp the images
        warped_ref = cv2.warpPerspective(ref, H0, (width, height))
        
        def histogram_matching_single_channel(warped_ref, target):
            src_hist, bins = np.histogram(warped_ref.flatten(), 256, [0, 256])
            tmpl_hist, bins = np.histogram(target.flatten(), 256, [0, 256])

            cdf_src = src_hist.cumsum()
            cdf_tmpl = tmpl_hist.cumsum()

            cdf_src = (cdf_src - cdf_src.min()) * 255 / (cdf_src.max() - cdf_src.min())
            cdf_tmpl = (cdf_tmpl - cdf_tmpl.min()) * 255 / (cdf_tmpl.max() - cdf_tmpl.min())

            cdf_src = np.ma.filled(np.ma.masked_equal(cdf_src, 0), 0).astype('uint8')
            cdf_tmpl = np.ma.filled(np.ma.masked_equal(cdf_tmpl, 0), 0).astype('uint8')

            mapping = np.zeros(256, dtype='uint8')
            for src_value in range(256):
                diff = abs(cdf_tmpl - cdf_src[src_value])
                mapping[src_value] = np.argmin(diff)

            matched = cv2.LUT(warped_ref, mapping)
            return matched

        if self.histogram_matching:
            return histogram_matching_single_channel(warped_ref, target).astype(np.uint8)
        
        else:
            return warped_ref.astype(np.uint8)