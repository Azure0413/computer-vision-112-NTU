import numpy as np
import cv2
import random
from tqdm import tqdm
from utils import solve_homography, warping

random.seed(999)

def panorama(imgs):
    """
    Image stitching with estimated homograpy between consecutive
    :param imgs: list of images to be stitched
    :return: stitched panorama
    """
    h_max = max([x.shape[0] for x in imgs])
    w_max = sum([x.shape[1] for x in imgs])

    # create the final stitched canvas
    dst = np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8)
    dst[:imgs[0].shape[0], :imgs[0].shape[1]] = imgs[0]
    last_best_H = np.eye(3)
    out = None

    # for all images to be stitched:
    for i in tqdm(range(len(imgs)-1)):
        im1 = imgs[i]
        im2 = imgs[i + 1]
        
        # TODO: 1.feature detection & matching
        orb = cv2.ORB_create()
        keypoint1, des1 = orb.detectAndCompute(im2, None)
        keypoint2, des2 = orb.detectAndCompute(im1, None)
        # matches = bf.knnMatch(des1, des2, k=2)        
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        n_best_matches = 100
        matches = sorted(matches, key = lambda x: x.distance)[:n_best_matches]
        q_idx = [match.queryIdx for match in matches]
        t_idx = [match.trainIdx for match in matches]
        src_pts = np.array([keypoint1[i].pt for i in q_idx])
        dst_pts = np.array([keypoint2[i].pt for i in t_idx])
        
        # TODO: 2. apply RANSAC to choose best H
        maxInliers = 0
        num_kps_for_H = 13
        threshold = 0.8
        times = 5000
        for _ in range(times):
            rand_idx = random.sample(range(len(src_pts)), num_kps_for_H)
            p1, p2 = src_pts[rand_idx], dst_pts[rand_idx]
            H = solve_homography(p1, p2)
            U = np.concatenate((src_pts.T, np.ones((1,src_pts.shape[0]))), axis=0)
            pred = np.dot(H, U)
            pred = (pred/pred[2]).T[:,:2]
            distance = pred-dst_pts
            error = np.linalg.norm(distance, axis=1)
            inliers = (error < threshold).sum()
            if inliers > maxInliers :
                best_H = H.copy()
                maxInliers = inliers
                
        # TODO: 3. chain the homographies
        last_best_H = np.dot(last_best_H, best_H)
        
        # TODO: 4. apply warping
        out = warping(im2, dst, last_best_H, 0, h_max, 0, w_max, direction='b')

    return out 

if __name__ == "__main__":
    # ================== Part 4: Panorama ========================
    # TODO: change the number of frames to be stitched
    FRAME_NUM = 3
    imgs = [cv2.imread('../resource/frame{:d}.jpg'.format(x)) for x in range(1, FRAME_NUM + 1)]
    output4 = panorama(imgs)
    cv2.imwrite('output4.png', output4)