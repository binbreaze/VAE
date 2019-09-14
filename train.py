# from root.test.ctlcloud.VAE import *
from sklearn.preprocessing import StandardScaler
from keras.models import Model,load_model
import pandas as pd
import pandas as pd
from keras.models import Model,load_model
import keras.backend as K
from  keras.losses import mean_absolute_error
from keras.layers import Dense,Input,Lambda,Layer
from keras   import regularizers
import numpy as np
# import matplotlib.pyplot as plt

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
    # return z_mean + K.exp(z_log_var) * epsilon
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
    model.compile(optimizer='adam', metrics=['loss', 'acc'])

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




x = pd.read_csv('D:\\taocloud\csvfile0911\cluster1_cpu_localhost.localdomain_cpu0_train.csv')
data = x['mean_usage_idle']
scaler = StandardScaler()
data1 = to_sequence(data,5)
data2 = scaler.fit_transform(data1)

train_dataset = data2


# test_dataset = data2[3000:]
# for i in range(10):
#     index = np.random.randint(0,1100)
#     for j in range(5):
#         a = np.random.randint(8, 10)
#         test_dataset[index][j] = a
# print(test_dataset.shape)
# test_dataset = test_dataset.tolist()
# test_dataset[100] = 10
# test_dataset[101] = 10
# test_dataset[102] = 12
# test_dataset[103] = 15
# test_dataset[104] = 16

# test_dataset[1100] = 11
# test_dataset[1101] = 12
# test_dataset[1202] = 13
# test_dataset[1303] = 14
# test_dataset[1404] = 15
#



# test_dataset = np.array(test_dataset)
input_shape = (train_dataset.shape[1],)

model = donut(input_shape,10,3,5)
model.summary()
history = model.fit(train_dataset,batch_size=200,epochs=100,verbose=1)

# input = Input(input_shape)
# z,z_mean,z_log_var = encoder(input,10,3)
# z_x = decoder(z,10,5)
# y = CustomVariationalLayer(z_mean,z_log_var)([input,z_x])
#
# vae = Model(input,y)
# vae.compile(optimizer='adam',loss=None,metrics=['acc','loss'])
# vae.summary()
#
# history = vae.fit(train_dataset,batch_size=200,epochs=100,verbose=1)
# import matplotlib.pyplot as plt
# #
# # plt.plot(history.history['loss'],label='loss')
# # plt.legend()
# # plt.show()
# vae.save('model_vae.h5')
model.save_weights('model_vae_cpu_mean_usage_user_09011.h5')

# vae = load_model('model_vae.h5',custom_objects={'CustomVariationalLayer': CustomVariationalLayer})
# model.load_weights('model_vae_cpu_mean_usage_user_0904.h5')
# pre = model.predict(test_dataset)
# print(pre.shape)
# test = scaler.inverse_transform(test_dataset)
# pred = scaler.inverse_transform(pre)
# # print(pre.shape)
# test_data_ = list()
# pre_ = list()
# # # abnormal = list()
# # abnormal_index = list()
# score = list()
# # num =0
# # mean_pre = np.mean(pre,axis=1).tolist()
# # std_pre = np.std(pre,axis=1).tolist()
# #
# # print(mean_pre)
# # print(std_pre)
#
#
# for i  in range(test.shape[0]):
#     test_data_.append(test[i][0])
#     pre_.append(pred[i][0])
#     cha = abs(test[i][0] - pred[i][0])
#
#     score.append(cha)
# mean_score = np.mean(score)
# seq_score = to_sequence(score,1)
# fazhi = mean_score * 5
# abnormal_index = list()
# for i  in range(seq_score.shape[0]):
#     if len(list(filter(lambda x : x > fazhi,seq_score[i]))) == len(seq_score[i]):
#         abnormal_index.append(i)
# print(abnormal_index)
#
#
# plt.plot(test_data_,label='test')
# plt.plot(pre_,label='pre')
# # plt.plot(score,label='score')
# # plt.title('zhouqi_data')
# # plt.savefig('./zhouqi_data.png')
# plt.legend()
# plt.show()
# plt.plot(score,label='score')
# plt.legend()
# plt.show()

# fazhi = max(score)*0.2
# fazhi_ = min(score)*0.2
#
# score_seq = to_sequence(score,5)
# abnormal = list()
# abnor = list()
# for i in range(score_seq.shape[0]):
#     if len(list(filter(lambda x : x>fazhi,score_seq[i]))) == len(score_seq[i]):
#         abnormal.append(i)
# for i in abnormal:
#     for j in range(i,i+5):
#         abnor.append(j)
# abnormal_index = list(set(abnor))
# no = list()
#
# for i in abnormal_index:
#     test_data_[i] = None
#
#
# print(abnormal_index)
# plt.plot(test_data_[:50],label='data')
# # plt.plot(pre_[:50],label='pre_')
# # plt.plot(pre_[200:300],label='pre')
# plt.plot(score[:50],label='score')
#
# plt.legend()
# plt.show()


#
# for i in range(len(score)-3):
#     if (score[i] > fazhi) and (score[i+1] > fazhi) and(score[i+2] > fazhi):
#             abnormal_.append(i)
#     elif (score[i] < fazhi_) and (score[i+1] < fazhi_) and(score[i+2] < fazhi_):
#             abnormal_.append(i)
#
# print(abnormal_)
#
#     # pre_sub.append(cha)
#     # if abs(cha) >= 0.3:
#     #     num += 1
#     #     # test_data_.append(None)
#     #     abnormal_index.append(i)
# plt.plot(test_data_,label='test')
# plt.plot(pre_,label='pre')
# # plt.plot(score,label='cha')
#
# plt.legend()
# plt.show()
#
# plt.plot(score[:50],label='score')
# plt.legend()
# plt.show()






# print(num)
# print(abnormal_index)
# test_data_ = np.array(test_data_)
# # pre_ = np.array(pre_)
# # test_data_max = np.max(test_data_,axis=0)
#
# # pre_max = np.max(pre_,axis=0)
#
#
# # print('最大值：{predata},{pre}'.format(predata=test_data_max,pre=pre_max))
# plt.plot(test_data_,markevery=[test[i] for i in abnormal_index])
#
# plt.legend()
# plt.show()

# plt.plot(pre_sub,label='pre_sub')
# plt.legend()
# plt.show()


