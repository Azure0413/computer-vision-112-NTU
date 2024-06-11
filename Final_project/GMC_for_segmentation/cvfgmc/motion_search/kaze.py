import cv2
import numpy as np
import h5py
from .base import FeatureModel

class KAZE_Motion(FeatureModel):
    def __init__(self, orb_params: dict=None, BFMatcher_params: dict=None):
        if orb_params is None:
            self.akaze_params = {"threshold": 0.00005, "nOctaves": 5, "nOctaveLayers": 5}
        if BFMatcher_params is None:
            self.BFMatcher_params = {"normType": cv2.NORM_L2, "crossCheck": True}

        self.feature_detector = cv2.KAZE_create(**self.akaze_params)
        self.BFMatcher = cv2.BFMatcher(**self.BFMatcher_params)