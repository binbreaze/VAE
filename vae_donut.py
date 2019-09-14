import pandas as pd
from keras.models import Model,load_model
import keras.backend as K
from  keras.losses import mean_absolute_error
from keras.layers import Dense,Input,Lambda
from keras   import regularizers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
       # Arguments:
           args (tensor): mean and log of variance of Q(z|X)
        c = (z_log_var + e_std ) * e_mean+  z_mean
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

def donut_model(input_shape,intermediate_dim,z_dim,x_dim):
    input = Input(input_shape,name='encoder_input')

    # encoder
    dense_z1 = Dense(intermediate_dim,
                    activation='relu',
                    kernel_regularizer=regularizers.l2(0.001))(input)
    dense_z2 = Dense(intermediate_dim,
                   activation='relu',
                   kernel_regularizer=regularizers.l2(0.001))(dense_z1)

    z_mean = Dense(z_dim,name='z_mean')(dense_z2)
    z_log_var = Dense(z_dim,activation='softplus',name='z_log_var')(dense_z2)

    z = Lambda(sampling,output_shape=(z_dim,),name='z')([z_mean,z_log_var])
    # decoder
    dense_x1 = Dense(intermediate_dim,
                     activation='relu',
                     kernel_regularizer=regularizers.l2(0.001))(z)
    dense_x2 = Dense(intermediate_dim,
                     activation='relu',
                     kernel_regularizer=regularizers.l2(0.001))(dense_x1)

    zx_mean = Dense(x_dim,name='zx_mean')(dense_x2)
    zx_log_var = Dense(x_dim,activation='softplus',name='zx_log_var')(dense_x2)

    z_x = Lambda(sampling,output_shape=(x_dim,),name='z_x')([zx_mean,zx_log_var])

    model = Model(input,z_x,name='vae_donut')

    reconstruction_loss = mean_absolute_error(input,z_x)
    reconstruction_loss *= x_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss,axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    model.add_loss(vae_loss)
    model.compile(optimizer='sgd', metrics=['loss', 'acc'])

    return model
# def vae_loss(y_true,y_pred):
#     reconstruction_loss =  mean_absolute_error(y_true,y_pred)

x = pd.read_csv('D:\\taocloud\donut-master\sample_data\cpu4.csv')
x = x['value']
scaler = StandardScaler()
x = to_sequence(x,5)
x = scaler.fit_transform(x)

train_dataset = x[:15000]
# val_x = x[10000:11000]
test_dataset = x[15000:]
# train_dataset = to_sequence(train_x,5)
# # val_dataset = to_sequence(val_x,5)
# test_dataset = to_sequence(test_x,5)
# train_dataset= scaler.fit_transform(train_dataset)

input_shape = (train_dataset.shape[1],)
model =  donut_model(input_shape,4,3,5)
model.summary()

#
history = model.fit(train_dataset,
          batch_size=100,
          epochs=500,
          # validation_data=val_dataset,
          verbose=1)
# model.save('vae_0823.h5')

model.save_weights('model_weight_0826.h5')
# model.load_weights('model_weight_0826.h5')
# plt.plot(history.history['loss'],label='loss')
# plt.legend()
# plt.show()

# model = load_model('./vae_0823.h5',custom_objects={'my_loss':donut_model(input_shape,4,3,5).loss})
pre  = model.predict(test_dataset)
# print(pre.shape)
test = scaler.inverse_transform(test_dataset)
pre = scaler.inverse_transform(pre)
test_data_ = list()
pre_ = list()
pre_sub = list()
for i  in range(test.shape[0]):
    test_data_.append(test[i][0])

    pre_.append(pre[i][0])
    pre_sub.append(abs(test[i][0] - pre[i][0]))
test_data_ = np.array(test_data_)
pre_ = np.array(pre_)
test_data_max = np.max(test_data_,axis=0)
pre_max = np.max(pre_,axis=0)

print('最大值：{predata},{pre}'.format(predata=test_data_max,pre=pre_max))
plt.plot(test_data_,label='test_data')
plt.plot(pre_,label='pre')

plt.legend()
plt.show()

plt.plot(pre_sub,label='pre_sub')
plt.legend()
plt.show()