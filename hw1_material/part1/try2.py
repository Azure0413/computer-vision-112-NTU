# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 17:45:45 2024

@author: Eric
"""

keypoints = np.array([],dtype='int64').reshape((0,2))
for i in range(self.num_octaves):
    images = np.array(dog_images[i])
    cube = np.array([np.roll(images,(x,y,z),axis=(2,1,0)) 
                     for z in range(-1,2) for y in range(-1,2) for x in range(-1,2)])
    masked = (np.absolute(images)>=self.threshold)&((np.min(cube,axis=0)==images)|(np.max(cube,axis=0)==images))
    for j in range(1, self.num_DoG_images_per_octave-1):
        m = masked[j]
        x, y = np.meshgrid(np.arange(m.shape[1]),np.arange(m.shape[0]))
        key_p = np.stack([y[m],x[m]]).T*2 if i else np.stack([y[m],x[m]]).T
        keypoints = np.concatenate([keypoints,key_p])