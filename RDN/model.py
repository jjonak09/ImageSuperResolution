import keras.backend as K
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, concatenate
from keras.layers import BatchNormalization, Activation, Add
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.activations import sigmoid,relu,softplus
from keras.initializers import RandomNormal

conv_init = RandomNormal(0, 0.02)


# def SrModel(input_shape):
#     input = Input(shape=input_shape)
#     h = Conv2D(64,9,strides=1,padding='same')(input)
#     h = Activation('relu')(h)
#     h = Conv2D(32,1,strides=1,padding='same')(h)
#     h = Activation('relu')(h)
#     h = Conv2D(3,5,strides=1,padding='same')(h)
#     h = Activation('relu')(h)
#     output = h
#     return Model(inputs=input,outputs=output)

# def Gen_ResBlock(layer_input,base):
#         x = Conv2D(base,3,strides=1,padding="same")(layer_input)
#         x = Activation('relu')(x)
#         x = Conv2D(base,3,strides=1,padding="same")(x)
#         return Add()([x,layer_input])
#
# def CBR(layer_input,out_channel=256):
#     x = UpSampling2D(size=2)(layer_input)
#     x = Conv2D(out_channel,3,strides=1,padding="same")(x)
#     x = Activation('relu')(x)
#     return x
#
#
# def SrModel(input_shape,base=64):
#
#     input = Input(shape=input_shape)
#     h = Conv2D(base,9,strides=1,padding="same")(input)
#     h1 = Activation('relu')(h)
#     h = Gen_ResBlock(h1,base=base)
#     h = Gen_ResBlock(h,base=base)
#     h = Gen_ResBlock(h,base=base)
#     h = Gen_ResBlock(h,base=base)
#     h2 = Conv2D(base,3,strides=1,padding="same")(h)
#     h = Add()([h2,h1])
#     h = CBR(h)
#     h = CBR(h)
#     output = Conv2D(3,9,strides=1,padding="same")(h)
#
#     return Model(inputs=input,outputs=output)

def RDB(layer_input, base):
    h1 = Conv2D(base, 3, strides=1, padding="same",kernel_initializer=conv_init)(layer_input)
    h1 = Activation('relu')(h1)
    c1 = concatenate([h1,layer_input],axis=3)
    h2 = Conv2D(base, 3, strides=1, padding="same",kernel_initializer=conv_init)(c1)
    h2 = Activation('relu')(h2)
    c2 = concatenate([h2,h1,layer_input],axis=3)
    h3 = Conv2D(base, 3, strides=1, padding="same",kernel_initializer=conv_init)(c2)
    h3 = Activation('relu')(h3)
    c3 = concatenate([h3,h2,h1,layer_input],axis=3)
    x = Conv2D(base, 1, strides=1, padding="same",kernel_initializer=conv_init)(c3)
    return Add()([x, layer_input])


def CBR(channels, layer_input):
    x = UpSampling2D(size=2)(layer_input)
    x = Conv2D(channels,3,strides=1,padding="same",kernel_initializer=conv_init)(x)
    return x


def SrModel(input_shape,base=64):
    input = Input(shape=(input_shape))
    s1 = Conv2D(base, 3, strides=1, padding="same",kernel_initializer=conv_init)(input)
    s2 = Conv2D(base, 3, strides=1, padding="same",kernel_initializer=conv_init)(s1)
    r1 = RDB(s2, base=base)
    r2 = RDB(r1, base=base)
    r3 = RDB(r2, base=base)
    r4 = RDB(r3, base=base)
    r5 = RDB(r4, base=base)
    r6 = RDB(r5, base=base)
    r7 = RDB(r6, base=base)
    r8 = RDB(r7, base=base)
    r9 = RDB(r8, base=base)
    r10 = RDB(r9, base=base)
    r11 = RDB(r10, base=base)
    r12 = RDB(r11, base=base)
    r13 = RDB(r12, base=base)
    concate = concatenate([r13,r12,r11,r10,r9,r8,r7,r6,r5,r4,r3,r2,r1],axis=3)
    h = Conv2D(base, 1, strides=1, padding="same",kernel_initializer=conv_init)(concate)
    h = Conv2D(base, 3, strides=1, padding="same",kernel_initializer=conv_init)(h)
    c2 = Add()([h, s1])
    u = CBR(base *4,c2)
    # u = CBR(base *4,u)
    output = Conv2D(3, 3, strides=1, padding="same",kernel_initializer=conv_init)(u)
    return Model(inputs=input, outputs=output)


if __name__ == '__main__':
    lr = (128,128,3)
    sr_model = SrModel(lr)
    sr_model.summary()
