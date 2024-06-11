import numpy as np
import cv2

from .base import MotionModel

class OptFlow(MotionModel):
    def __init__(self, method: str='foward'):
        self.method = method

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
        # Ensure that the input images are grayscale
        if len(ref0.shape) == 3:
            ref0 = cv2.cvtColor(ref0, cv2.COLOR_BGR2GRAY)
        if len(target.shape) == 3:
            target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
        
        # Ensure that the input images have the same dimensions
        if ref0.shape != target.shape:
            raise ValueError("Input images must have the same dimensions")
        
        # Calculate dense optical flow from ref0 to target
        flow = cv2.calcOpticalFlowFarneback(ref0, target, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Generate grid for remapping
        h, w = ref0.shape[:2]
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Compute the intermediate frame coordinates for ref0
        intermediate_x = grid_x + flow[..., 0]
        intermediate_y = grid_y + flow[..., 1]
        
        # Warp ref0 towards target using the flow vectors
        compensated = cv2.remap(ref0, intermediate_x.astype(np.float32), intermediate_y.astype(np.float32), cv2.INTER_LINEAR)
        
        return compensated