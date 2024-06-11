import numpy as np
import cv2

class Difference_of_Gaussian(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.sigma = 2**(1/4)
        self.num_octaves = 2
        self.num_DoG_images_per_octave = 4
        self.num_guassian_images_per_octave = self.num_DoG_images_per_octave + 1

    def get_keypoints(self, image):
        # Step 1: Filter images with different sigma values (5 images per octave, 2 octave in total)
        # - Function: cv2.GaussianBlur (kernel = (0, 0), sigma = self.sigma**___)
        origin_octave = [image]
        for i in range(1, self.num_guassian_images_per_octave):
            blurred_image = cv2.GaussianBlur(image, (0, 0), self.sigma**i)
            origin_octave.append(blurred_image)
        
        DSImage = cv2.resize(origin_octave[-1], (image.shape[1]//2, image.shape[0]//2), interpolation = cv2.INTER_NEAREST)
        resize_octave = [DSImage]
        for i in range(1, self.num_guassian_images_per_octave):
            blurred_image = cv2.GaussianBlur(DSImage, (0, 0), self.sigma**i)
            resize_octave.append(blurred_image)

        gaussian_images = [origin_octave, resize_octave]

        # Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
        # - Function: cv2.subtract(second_image, first_image)
        dog_images = []
        for i in range(self.num_octaves):
            image = gaussian_images[i]
            dog_image = []
            for j in range(self.num_DoG_images_per_octave):
                difference = cv2.subtract(image[j+1], image[j])
                dog_image.append(difference)
                normalize = (difference-min(difference.flatten()))*255/(max(difference.flatten())-min(difference.flatten()))
                cv2.imwrite(f'testdata/DoG{i+1}-{j+1}.png', normalize)
            dog_images.append(dog_image)

        # Step 3: Thresholding the value and Find local extremum (local maximum and local minimum)
        #         Keep local extremum as a keypoint
        keypoints = []
        for i in range(self.num_octaves):
            images = np.array(dog_images[i])
            height, width = images[i].shape
            for j in range(1, self.num_DoG_images_per_octave-1):
                for x in range(1, width-2):
                    for y in range(1, height-2):
                        pixel = images[j,y,x]
                        cube = images[j-1:j+2, y-1:y+2, x-1:x+2]
                        if (np.absolute(pixel) > self.threshold) and ((pixel >= cube).all() or (pixel <= cube).all()):
                            keypoints.append([y*2, x*2] if i else [y, x])

        # Step 4: Delete duplicate keypoints
        # - Function: np.unique
        keypoints = np.unique(np.array(keypoints), axis = 0)

        # Sort keypoints by y, then by x
        keypoints = keypoints[np.lexsort((keypoints[:,1],keypoints[:,0]))] 
        return keypoints