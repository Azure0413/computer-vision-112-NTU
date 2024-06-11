import cv2
import numpy as np
import h5py
from .base import FeatureModel

class AKAZE_Motion(FeatureModel):
    def __init__(self, akaze_params: dict=None, BFMatcher_params: dict=None):
        if akaze_params is None:
            self.akaze_params = {"threshold": 0.00005, "nOctaves": 5, "nOctaveLayers": 5, "max_points": 200000, "descriptor_type": 2}
        if BFMatcher_params is None:
            self.BFMatcher_params = {"normType": cv2.NORM_L2, "crossCheck": True}
        self.feature_detector = cv2.AKAZE_create(**self.akaze_params)
        self.BFMatcher = cv2.BFMatcher(**self.BFMatcher_params)