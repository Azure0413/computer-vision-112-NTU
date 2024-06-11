# Models
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
import ants


# 運動估計
def affine_registration(ref1_frame, ref2_frame):
    ref1_ants = ants.from_numpy(ref1_frame.astype(np.float32))
    ref2_ants = ants.from_numpy(ref2_frame.astype(np.float32))
    
    transform = ants.registration(fixed=ref1_ants, moving=ref2_ants, type_of_transform='Affine')
    
    matrix = transform['fwdtransforms'][0]
    
    if matrix is None:
        return ref1_frame

    matrix = ants.read_transform(matrix).parameters
    matrix = np.array(matrix).reshape(2, 3)
    
    height, width = ref1_frame.shape[:2]
    
    compensated_frame = cv2.warpAffine(ref1_frame, matrix, (width, height))
    
    return compensated_frame

# 直接映射
def block_motion_compensation_affine(ref1_frame, ref2_frame):
    height, width = ref1_frame.shape
    matrix = cv2.getAffineTransform(np.float32([[0,0],[width-1,0],[0,height-1]]), 
                                    np.float32([[0,0],[width-1,0],[0,height-1]]))
    compensated_frame = cv2.warpAffine(ref1_frame, matrix, (width, height))
    return compensated_frame

# ORB特徵點估計
def block_motion_compensation_feature_matching(ref1_frame, ref2_frame):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(ref1_frame, None)
    kp2, des2 = orb.detectAndCompute(ref2_frame, None)
    
    if des1 is None or des2 is None:
        return ref1_frame
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if len(matches) < 4:
        return ref1_frame
    
    matches = sorted(matches, key=lambda x: x.distance)
    
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    matrix, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    
    if matrix is None:
        return ref1_frame

    height, width = ref1_frame.shape
    compensated_frame = cv2.warpAffine(ref1_frame, matrix, (width, height))
    
    return compensated_frame

def block_motion_compensation_feature_matching_perspective(ref1_frame, ref2_frame):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(ref1_frame, None)
    kp2, des2 = orb.detectAndCompute(ref2_frame, None)
    
    if des1 is None or des2 is None:
        return ref1_frame
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if len(matches) < 4:
        return ref1_frame
    
    matches = sorted(matches, key=lambda x: x.distance)
    
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
    
    if matrix is None:
        return ref1_frame

    height, width = ref1_frame.shape
    compensated_frame = cv2.warpPerspective(ref1_frame, matrix, (width, height))
    
    return compensated_frame

# 相位轉換(離散傅里葉變換)
def block_motion_compensation_phase_correlation(ref1_frame, ref2_frame):
    ref1_float = ref1_frame.astype(np.float32)
    ref2_float = ref2_frame.astype(np.float32)
    
    dft1 = cv2.dft(ref1_float, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft2 = cv2.dft(ref2_float, flags=cv2.DFT_COMPLEX_OUTPUT)
    
    cross_power_spectrum = dft1 * np.conj(dft2)
    abs_cross_power_spectrum = np.abs(cross_power_spectrum)
    abs_cross_power_spectrum[abs_cross_power_spectrum == 0] = np.finfo(float).eps
    
    cross_power_spectrum /= abs_cross_power_spectrum
    cross_corr = cv2.idft(cross_power_spectrum, flags=cv2.DFT_REAL_OUTPUT | cv2.DFT_SCALE)
    _, _, max_loc, _ = cv2.minMaxLoc(cross_corr)
    
    dx, dy = max_loc
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    height, width = ref1_frame.shape
    compensated_frame = cv2.warpAffine(ref1_frame, matrix, (width, height))
    
    return compensated_frame

# AKAZE特徵點估計
def AKAZE_feature_compensation(ref1_frame, ref2_frame):
    akaze = cv2.AKAZE_create()
    kp1, des1 = akaze.detectAndCompute(ref1_frame, None)
    kp2, des2 = akaze.detectAndCompute(ref2_frame, None)
    
    if des1 is None or des2 is None:
        return ref1_frame
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    
    if len(good_matches) < 4:
        return ref1_frame
    
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    matrix, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    
    if matrix is None:
        return ref1_frame
    
    height, width = ref1_frame.shape
    compensated_frame = cv2.warpAffine(ref1_frame, matrix, (width, height))
    
    return compensated_frame

def AKAZE_feature_compensation_perspective(ref1_frame, ref2_frame):
    akaze = cv2.AKAZE_create()
    kp1, des1 = akaze.detectAndCompute(ref1_frame, None)
    kp2, des2 = akaze.detectAndCompute(ref2_frame, None)
    
    if des1 is None or des2 is None:
        return ref1_frame
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    
    if len(good_matches) < 4:
        return ref1_frame
    
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
    
    if matrix is None:
        return ref1_frame

    height, width = ref1_frame.shape
    compensated_frame = cv2.warpPerspective(ref1_frame, matrix, (width, height))
    
    return compensated_frame

# kanade光流法
def block_motion_compensation_lucas_kanade(ref1_frame, ref2_frame):
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

# SIFT特徵點估計
def block_motion_compensation_sift(ref1_frame, ref2_frame):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(ref1_frame, None)
    kp2, des2 = sift.detectAndCompute(ref2_frame, None)
    
    if des1 is None or des2 is None:
        return ref1_frame
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
            
    if len(good_matches) < 4:
        return ref1_frame
    
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    matrix, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    
    if matrix is None:
        return ref1_frame

    height, width = ref1_frame.shape
    compensated_frame = cv2.warpAffine(ref1_frame, matrix, (width, height))
    
    return compensated_frame

def block_perspective_sift(ref1_frame, ref2_frame):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(ref1_frame, None)
    kp2, des2 = sift.detectAndCompute(ref2_frame, None)
    
    if des1 is None or des2 is None:
        return ref1_frame
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
            
    if len(good_matches) < 4:
        return ref1_frame
    
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
    
    if matrix is None:
        return ref1_frame

    height, width = ref1_frame.shape
    compensated_frame = cv2.warpPerspective(ref1_frame, matrix, (width, height))
    
    return compensated_frame

def block_motion_compensation_block_matching(ref1_frame, ref2_frame):
    result = cv2.matchTemplate(ref2_frame, ref1_frame, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(result)
    dx, dy = max_loc
    height, width = ref1_frame.shape
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    compensated_frame = cv2.warpAffine(ref1_frame, matrix, (width, height))
    return compensated_frame

def block_motion_compensation_affine(ref1_frame, ref2_frame):
    height, width = ref1_frame.shape
    matrix = cv2.getAffineTransform(np.float32([[0,0],[width-1,0],[0,height-1]]), 
                                    np.float32([[0,0],[width-1,0],[0,height-1]]))
    compensated_frame = cv2.warpAffine(ref1_frame, matrix, (width, height))
    return compensated_frame