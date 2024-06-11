import numpy as np
from .base import MotionModel

class Naive(MotionModel):
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
        if self.method == 'forward':
            return ref0
        elif self.method == 'backward':
            return ref1
        else:
            return target
        
    