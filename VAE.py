import pandas as pd
from keras.models import Model,load_model
import keras.backend as K
from  keras.losses import mean_absolute_error
from keras.layers import Dense,Input,Lambda,Layer
from keras   import regularizers
import numpy as np
import matplotlib.pyplot as plt

def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
       # Arguments:
           args (tensor): mean and log of variance of Q(z|X)
       # Returns:
           z (tensor): sampled latent vector
       """
    z_mean,z_log_var = args
    batch  = K.shape(z_mean)[0]
    dim = K.shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch,dim),
                              mean=0.0,stddev=1.0)
    std_epsilon = 1e-4
    return z_mean + (z_log_var + std_epsilon) * epsilon

def to_sequence(data,n_input):
    n_start = 0
    data_len = len(data)
    data_x  = list()
    data_x_len = data_len - n_input + 1
    if n_start < data_x_len:
        for _ in range(data_x_len):
            n_end = n_start + n_input
            data_x.append(data[n_start:n_end])
            n_start += 1
    return np.array(data_x)

def input(input_shape):
    return Input(input_shape,name='input')

def encoder(input,intermediate_dim,z_dim):
    # encoder
    dense_z1 = Dense(intermediate_dim,
                     activation='relu',
                     kernel_regularizer=regularizers.l2(0.001))(input)
    dense_z2 = Dense(intermediate_dim,
                     activation='relu',
                     kernel_regularizer=regularizers.l2(0.001))(dense_z1)

    z_mean = Dense(z_dim, name='z_mean')(dense_z2)
    z_log_var = Dense(z_dim, activation='softplus', name='z_log_var')(dense_z2)

    z = Lambda(sampling, output_shape=(z_dim,), name='z')([z_mean,z_log_var])
    return z,z_mean,z_log_var
def decoder(z,intermediate_dim,x_dim):
    # decoder
    dense_x1 = Dense(intermediate_dim,
                     activation='relu',
                     kernel_regularizer=regularizers.l2(0.001))(z)
    dense_x2 = Dense(intermediate_dim,
                     activation='relu',
                     kernel_regularizer=regularizers.l2(0.001))(dense_x1)

    x_mean = Dense(x_dim, name='zx_mean')(dense_x2)
    x_log_var = Dense(x_dim, activation='softplus', name='zx_log_var')(dense_x2)

    z_x = Lambda(sampling, output_shape=(x_dim,), name='z_x')([x_mean,x_log_var])
    return z_x

def donut(input_shape,intermediate_dim,z_dim,x_dim):
    inp = input(input_shape)
    z , z_mean, z_log_var = encoder(inp,intermediate_dim,z_dim)
    x = decoder(z,intermediate_dim,x_dim)
    model = Model(inp,x)
    reconstruction_loss = mean_absolute_error(inp,x)
    reconstruction_loss *= x_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    model.add_loss(vae_loss)
    model.compile(optimizer='sgd', metrics=['loss', 'acc'])

    return model

#
# class CustomVariationalLayer(Layer):
#     def __init__(self,z_mean,z_log_var,**kwargs):
#         self.is_placeholder = True
#         self.z_mean = z_mean
#         self.z_log_var = z_log_var
#         super(CustomVariationalLayer, self).__init__(**kwargs)
#
#     def vae_loss(self,x,z_x,z_mean,z_log_var):
#         mse = mean_absolute_error(x,z_x)
#         kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var),axis=-1)
#         return K.mean(mse + kl_loss)
#
#     def call(self, inputs,**kwargs):
#         x = inputs[0]
#         z_x = inputs[1]
#         loss = self.vae_loss(x,z_x,self.z_mean,self.z_log_var)
#         self.add_loss(loss,inputs=inputs)
#         return x


