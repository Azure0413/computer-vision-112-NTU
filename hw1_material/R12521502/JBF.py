import numpy as np
import cv2

class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6*sigma_s+1
        self.pad_w = 3*sigma_s
    
    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        
        ### TODO ###
        spatial_table = np.exp(-0.5*(np.arange(self.pad_w+1)**2)/self.sigma_s**2)
        range_table = np.exp(-0.5*(np.arange(256)/255)**2/self.sigma_r**2)
        wight = np.zeros(padded_img.shape)
        result = np.zeros(padded_img.shape)
        
        for x in range(-self.pad_w, self.pad_w+1):
            for y in range(-self.pad_w, self.pad_w+1):
                difference = range_table[np.abs(np.roll(padded_guidance, [y,x], axis=[0,1])-padded_guidance)]
                if difference.ndim==2:
                    r_wight = difference
                else:
                    r_wight = np.prod(difference,axis=2)
                s_wight = spatial_table[np.abs(x)]*spatial_table[np.abs(y)]
                t_wight = s_wight * r_wight
                padded_img_roll = np.roll(padded_img, [y,x], axis=[0,1])
                for i in range(padded_img.ndim):
                    result[:,:,i] += padded_img_roll[:,:,i]*t_wight
                    wight[:,:,i] += t_wight
        output = (result/wight)[self.pad_w:-self.pad_w, self.pad_w:-self.pad_w,:]
        return np.clip(output, 0, 255).astype(np.uint8)