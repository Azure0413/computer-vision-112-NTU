import cv2
import numpy as np
import h5py
from .base import FeatureModel

class SIFT_Motion(FeatureModel):
    def __init__(self, orb_params: dict=None, BFMatcher_params: dict=None):
        if orb_params is None:
            self.sift_params = {"nfeatures": 200000, "contrastThreshold": 0.03, "edgeThreshold": 35, "sigma": 1.6}
        if BFMatcher_params is None:
            self.BFMatcher_params = {"normType": cv2.NORM_L2, "crossCheck": True}

        self.feature_detector = cv2.SIFT_create(**self.sift_params)
        self.BFMatcher = cv2.BFMatcher(**self.BFMatcher_params)