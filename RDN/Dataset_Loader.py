import numpy as np
from PIL import Image
from glob import glob
import cv2


def Data_Loader(batchsize=1,train =True,dataset='real'):

    if train:
        path = glob(dataset + '_train/*.jpg')
    else:
        path = glob(dataset + '_test/*.jpg')
    batch_images = np.random.choice(path, size=batchsize)
    hr_images = []
    lr_images = []

    for index, image_path in enumerate(batch_images):
        img = Image.open(image_path)
        img = img.convert('RGB')
        img = np.array(img).astype(np.float)
        # h,w = img.shape[0],img.shape[1]
        # h_cut = np.random.randint(h - 258)
        # w_cut = np.random.randint(w - 258)
        #
        # img = img[h_cut:h_cut + 256, w_cut:w_cut + 256]

        hr_image = cv2.resize(img,(256,256),interpolation=cv2.INTER_CUBIC)
        lr_image = cv2.resize(img,(128,128),interpolation=cv2.INTER_CUBIC)

        hr_images.append(hr_image)
        lr_images.append(lr_image)

    hr_images = np.array(hr_images) / 127.5 - 1
    lr_images = np.array(lr_images) / 127.5 - 1

    return hr_images, lr_images
