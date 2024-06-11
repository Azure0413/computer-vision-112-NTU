from .base import MotionModel
import numpy as np


class Affline(MotionModel):
    def __init__(self):
        # If the dimension of window_size is 1, it will be converted to (window_size, window_size)
        pass

    def compensate(self, target: np.ndarray, ref0: np.ndarray, ref1: np.ndarray) -> np.ndarray:
        pass