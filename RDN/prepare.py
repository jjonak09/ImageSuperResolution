import cv2
import os
from glob import glob
import numpy as np

file_path = glob('DIV2K_train_HR/*.png')
count = 0

for image_path in file_path:
    image = cv2.imread(image_path)
    h,w = image.shape[0],image.shape[1]
    for i in range(10):
        r1 = np.random.randint(h - 258)
        r2 = np.random.randint(w - 258)

        crop_image = image[r1:r1 + 256,r2:r2 + 256]
        cv2.imwrite('real_train/' +str(count)+ '.jpg',crop_image)
        count += 1
