import cv2
import numpy as np
import h5py
from .base import FeatureModel

class ORB_Motion(FeatureModel):
    def __init__(self, orb_params: dict=None, BFMatcher_params: dict=None):
        if orb_params is None:
            self.orb_params = {"nfeatures": 200000, "WTA_K": 4}
        if BFMatcher_params is None:
            self.BFMatcher_params = {"normType": cv2.NORM_HAMMING2, "crossCheck": True}

        self.feature_detector = cv2.ORB_create(**self.orb_params)
        self.BFMatcher = cv2.BFMatcher(**self.BFMatcher_params)