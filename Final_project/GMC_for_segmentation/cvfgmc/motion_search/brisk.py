import cv2
import numpy as np
import h5py
from .base import FeatureModel

class BRISK_Motion(FeatureModel):
    def __init__(self, orb_params: dict=None, BFMatcher_params: dict=None):
        if orb_params is None:
            self.brisk_params = {"thresh": 15, "octaves": 4, "patternScale": 1.0}
        if BFMatcher_params is None:
            self.BFMatcher_params = {"normType": cv2.NORM_HAMMING, "crossCheck": True}

        self.feature_detector = cv2.BRISK_create(**self.brisk_params)
        self.BFMatcher = cv2.BFMatcher(**self.BFMatcher_params)