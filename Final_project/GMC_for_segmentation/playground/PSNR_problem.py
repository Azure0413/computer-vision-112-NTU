import numpy as np
import cv2 

target = np.random.randint(0, 255, (16, 16)).astype(np.uint8)
truth = np.random.rand(16, 16).astype(np.uint8)


def student(target, truth):
    print((target.astype(np.int32) - truth.astype(np.int32))**2)
    print(np.sum((target.astype(np.int16) - truth.astype(np.int16))**2)/target.size)

def example(target, truth):
    print((target - truth)**2)
    print(np.sum((target - truth)**2)/target.size)

if __name__ == "__main__":
    student(target, truth)
    example(target, truth)