'''This script demonstrates how to build a variational autoencoder with Keras.
 #Reference
 - Auto-Encoding Variational Bayes
   https://arxiv.org/abs/1312.6114
'''
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist
from Hcluster import hcluster
import pickle

with open("realVec.txt", 'rb') as savefile:    
    realVec = pickle.load(savefile)
with open("virVec.txt", 'rb') as savefile:    
    virVec = pickle.load(savefile)
inputVec = []
for i in range(len(realVec)):
    inputVec.append([])
    for item in realVec[i]:
        inputVec[i].append(item)
    for item in virVec[i]:
        inputVec[i].append(item)
# inputVec is the combination of three kinds of information
inputVec = np.array(inputVec)
print(inputVec[0])
print(realVec[0])
print(virVec[0])



# VAE parameters
batch_size = 32
original_dim = len(inputVec[0])
latent_dim = 2
intermediate_dim = 30
epochs = 500
epsilon_std = 1.0

print(original_dim)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)


# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

# instantiate VAE model
vae = Model(x, x_decoded_mean)

# Compute VAE loss
xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(xent_loss + kl_loss)

vae.add_loss(vae_loss)
vae.compile(optimizer='rmsprop')
vae.summary()


# train the VAE
vae.fit(inputVec,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size)

# build a model to project inputs on the latent space
encoder = Model(x, h)

# result 
inputVec_encoded = encoder.predict(inputVec, batch_size=batch_size)
with open("finalVec.txt", 'wb') as savefile:    
    pickle.dump(inputVec_encoded,savefile)

print(len(inputVec_encoded))
print(inputVec_encoded[0])
k, l = hcluster(inputVec_encoded, 10)
print(l)
for subl in l:
    subl.sort()
for i in range(len(l)):
    if len(l[i]) >= 1:
        for j in range(len(l[i])):
            print("type: {} doc: {}".format( (l[i][j] % 3),l[i][j] ))
            #print("type: {} doc: {} topic: {}".format( (l[i][j] % 3),l[i][j], label[l[i][j]]))
    print('\n')






