from keras import backend as K
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation,Add
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import  Model
from keras.activations import sigmoid,relu,softplus
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import os,sys
import numpy as np
import argparse
from Dataset_Loader import Data_Loader
from model import SrModel

# def psnr(y_true,y_pred):
#     return -10*K.log(
#         K.mean(K.flatten((y_true - y_pred))**2)
#     )/np.log(10)


# print用
def PSNRLossnp(y_true,y_pred):
        return -10* np.log(np.mean(np.square(y_pred - y_true)))/np.log(10)

def PSNRLoss(y_true, y_pred):
        return 10* K.log(255**2 /(K.mean(K.square(y_pred - y_true))))


def SSIM( y_true,y_pred):
    u_true = K.mean(y_true)
    u_pred = K.mean(y_pred)
    var_true = K.var(y_true)
    var_pred = K.var(y_pred)
    std_true = K.sqrt(var_true)
    std_pred = K.sqrt(var_pred)
    c1 = K.square(0.01*7)
    c2 = K.square(0.03*7)
    ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
    denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
    return ssim / denom



parser = argparse.ArgumentParser(description="test")
parser.add_argument("--epoch", default=100000, type=int,
                    help="the number of epochs")
parser.add_argument("--save_interval", default=100, type=int,
                    help="the interval of snapshot")
parser.add_argument("--model_interval", default=1000, type=int,
                    help="the interval of savemodel")
parser.add_argument("--batchsize", default=4, type=int, help="batch size")
parser.add_argument("--dataset", default='real', 
                    help="type of dataset")

args = parser.parse_args()


LR_SHAPE = (None,None,3)
HR_SHAPE = (None,None,3)
BATCH_SIZE = args.batchsize
EPOCHS = args.epoch
SAVE_INTERVAL = args.save_interval
SAVE_MODEL = args.model_interval
dataset = args.dataset

sr_model = SrModel(LR_SHAPE)

lr_image = Input(shape=LR_SHAPE)
hr_image = Input(shape=HR_SHAPE)
sr_image = sr_model(lr_image)

# loss = K.mean(K.flatten((sr_image - hr_image))**2)

# L1ロス
loss = K.mean(K.abs(sr_image - hr_image))

training_updates = Adam(lr=1e-4,beta_1=0.9, beta_2=0.999).get_updates(
    sr_model.trainable_weights,[PSNRLoss,SSIM],loss)

sr_train = K.function([lr_image,hr_image],
                        [loss],
                        training_updates)

for epoch in range(EPOCHS):
    hr_images, lr_images = Data_Loader(batchsize=BATCH_SIZE,dataset=dataset)
    err, = sr_train([lr_images,hr_images])
    print("epoch: {}, err: {}".format(epoch,err))

    if epoch%SAVE_INTERVAL == 0:
        r, c = 2, 3
        hr_images, lr_images = Data_Loader(batchsize=2,train=False,dataset=dataset)
        sr = sr_model.predict(lr_images)
        lr_images = 0.5 * lr_images + 0.5
        sr = 0.5 * sr + 0.5
        hr_images = 0.5 * hr_images + 0.5
        titles = ['Low resolution', 'Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for row in range(r):
            for col, image in enumerate([lr_images,sr, hr_images]):
                axs[row, col].imshow(image[row])
                axs[row, col].set_title(titles[col])
                axs[row, col].axis('off')
            cnt += 1
        fig.savefig("result/srgan-{}.jpg".format(epoch))
        plt.close()

        hr, lr = Data_Loader(batchsize=1,train=False,dataset=dataset)
        sr = sr_model.predict(lr)
        print("PSNR: {}".format(PSNRLossnp(sr,hr)))

        if epoch % SAVE_MODEL == 0:
            sr_model.save('sr_model/sr_' + dataset + '_{}_epoch.h5'.format(epoch))
