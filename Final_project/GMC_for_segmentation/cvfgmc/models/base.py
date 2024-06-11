import numpy as np

class MotionModel():
    def __init__(self):
        raise NotImplementedError("This is an abstract class. Please use a concrete implementation.")
    
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
        raise NotImplementedError("This is an abstract class. Please use a concrete implementation.")