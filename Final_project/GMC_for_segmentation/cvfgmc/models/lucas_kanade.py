
import numpy as np
from .base import MotionModel

class LucasKanade(MotionModel):
    def __init__(self, method: str = 'forward'):
        '''
        This class is a naive model that compensates the target frame directly with the reference frame.
        You can choose to compensate the target frame with the reference frame 0 or 1 by setting the method parameter.

        params:
        -------
            method: str
                'forward' or 'backward'. If 'forward', the reference frame is ref0. If 'backward', the reference frame is ref1
        '''
        
        if method not in ['forward', 'backward', 'exact']:
            raise ValueError(f"Invalid method. Expected 'forward' or 'backward', got {method}")
        self.method = method

    def compensate(self, target: np.ndarray, ref0: np.ndarray, ref1: np.ndarray) -> np.ndarray:
        lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        feature_params = dict(maxCorners=100, 
                        qualityLevel=0.3, 
                        minDistance=7, 
                        blockSize=7, 
                        useHarrisDetector=False, 
                        k=0.04)
        p0 = cv2.goodFeaturesToTrack(ref1_frame, mask=None, **feature_params)
        p1, st, err = cv2.calcOpticalFlowPyrLK(ref1_frame, ref2_frame, p0, None, **lk_params)
        
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        
        matrix, mask = cv2.estimateAffinePartial2D(good_old, good_new)
        
        if matrix is None:
            return ref1_frame
        
        height, width = ref1_frame.shape
        compensated_frame = cv2.warpAffine(ref1_frame, matrix, (width, height))
        
        return compensated_frame

    def _post_processing(self, taregt, compensated):
        return compensated
    