from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

'''Example of VAE on MNIST dataset using CNN

The VAE has a modular design. The encoder, decoder and VAE
are 3 models that share weights. After training the VAE model,
the encoder can be used to  generate latent vectors.
The decoder can be used to generate MNIST digits by sampling the
latent vector from a Gaussian distribution with mean=0 and std=1.

# Reference

[1] Kingma, Diederik P., and Max Welling.
"Auto-encoding variational bayes."
https://arxiv.org/abs/1312.6114

'''


'''This model is based on
https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder_deconv.py

The docstring at the top is the original docstring for the module.

I've changed this in a few ways including:

1. Changed to load GM instead of MNIST
2. Added control of many hyperparameters
3. Separated out loss function into parts
4. Moved creation of the models into a function so we can import safely
5. Created a util module for reusable stuff.


I run on ICHEC. Access interactively using:

srun -p GpuQ -N 1 -A nuig02 -t 1:00:00 --pty bash

Then on the new node:

module load cuda/10.0 
module load conda/2
source activate tf

'''







import numpy as np
np.random.seed(0) # for reproducibility
import matplotlib.pyplot as plt
import argparse
import os

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.layers import Conv2D, Flatten, Lambda
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras import backend as K

from util import print_drums, Gauss, Clip, load_GM_data, load_model



# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# then z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon




def make_model(input_shape, args):

    # network parameters
    kernel_size = args.kernel_size0, args.kernel_size1
    strides = args.stride0, args.stride1
    filters = args.filters
    dense_layer_size = args.dense_layer_size
    latent_dim = args.latent_dim
    dropout_rate = args.dropout_rate
    conv_layers = args.conv_layers
    filter_expansion = args.filter_expansion
    

    # VAE model = encoder + decoder
    # build encoder model
    inputs = Input(shape=input_shape, name='encoder_input')
    x = inputs
    for i in range(conv_layers):
        filters *= filter_expansion
        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   activation='relu',
                   strides=strides,
                   padding='same')(x)
        x = Dropout(rate=dropout_rate)(x)

    # shape info needed to build decoder model
    shape = K.int_shape(x)

    # generate latent vector Q(z|X)
    x = Flatten()(x)
    x = Dense(dense_layer_size, activation='relu')(x)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    #encoder.summary()
    #plot_model(encoder, to_file='vae_conv_keras_drums_architecture_encoder.pdf', show_shapes=True)

    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
    x = Reshape((shape[1], shape[2], shape[3]))(x)

    for i in range(conv_layers):
        x = Conv2DTranspose(filters=filters,
                            kernel_size=kernel_size,
                            activation='relu',
                            strides=strides,
                            padding='same')(x)
        x = Dropout(rate=dropout_rate)(x)
        filters //= filter_expansion

    outputs = Conv2DTranspose(filters=1,
                              kernel_size=kernel_size,
                              activation='sigmoid',
                              padding='same',
                              name='decoder_output')(x)

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    #decoder.summary()
    #plot_model(decoder, to_file='vae_conv_keras_drums_decoder.pdf', show_shapes=True)

    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae')

    return inputs, outputs, z_mean, z_log_var, encoder, decoder, vae

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "Which loss function, MSE or BCE (default)"
    parser.add_argument("-l", "--loss", help=help_, default="BCE", type=str)
    help_ = "Latent dimension"
    parser.add_argument("-z", "--latent_dim", default=2, type=int)
    help_ = "Batch size"
    parser.add_argument("-b", "--batch_size", default=128, type=int)
    help_ = "Kernel size 0"
    parser.add_argument("--kernel_size0", default=9, type=int)
    help_ = "Kernel size 1"
    parser.add_argument("--kernel_size1", default=3, type=int)
    help_ = "Stride 0"
    parser.add_argument("--stride0", default=1, type=int)
    help_ = "Stride 1"
    parser.add_argument("--stride1", default=1, type=int)
    help_ = "Filters"
    parser.add_argument("--filters", default=16, type=int)
    help_ = "Dense layer size"
    parser.add_argument("--dense_layer_size", default=16, type=int)
    help_ = "Epochs"
    parser.add_argument("--epochs", default=100, type=int)
    help_ = "Lambda"
    parser.add_argument("--Lambda", default=0.01, type=float)
    help_ = "Dropout rate"
    parser.add_argument("--dropout_rate", default=0.2, type=float)
    help_ = "Conv layers"
    parser.add_argument("--conv_layers", default=2, type=int)
    help_ = "Filter expansion"
    parser.add_argument("--filter_expansion", default=2, type=int)
    
    
    
    args = parser.parse_args()
    print(args)
    open("vae_conv_keras_drums_args.txt", "w").write(str(args))

    filename = "../data/Groove_Monkee_Mega_Pack_GM.npy"
    (x_train, y_train), (x_test, y_test) = load_GM_data(filename)

    image_size = x_train.shape[1], x_train.shape[2]
    x_train = np.reshape(x_train, [-1, image_size[0], image_size[1], 1])
    x_test = np.reshape(x_test, [-1, image_size[0], image_size[1], 1])
    input_shape = (image_size[0], image_size[1], 1)


    inputs, outputs, z_mean, z_log_var, encoder, decoder, vae = make_model(input_shape, args)

    models = (encoder, decoder)
    data = (x_test, y_test)

    epochs = args.epochs
    Lambda = args.Lambda
    batch_size = args.batch_size

    
    

    def recon_loss(y_true, y_pred): # dummy variables
        # VAE loss = mse_loss or xent_loss + kl_loss
        if args.loss == "MSE":
            reconstruction_loss = mse(K.flatten(inputs), K.flatten(outputs))
        elif args.loss == "BCE":
            reconstruction_loss = binary_crossentropy(K.flatten(inputs),
                                                      K.flatten(outputs))
        else:
            raise ValueError

        # multiply by n data points, divide by batch size:
        # https://www.quora.com/How-do-you-fix-a-Variational-Autoencoder-VAE-that-suffers-from-mode-collapse
        reconstruction_loss *= (x_train.shape[0] / batch_size)
        return reconstruction_loss

    def KL_loss(y_true, y_pred): # dummy variables
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        return kl_loss

    def VAE_loss(y_true, y_pred):
        vae_loss = K.mean(recon_loss(y_true, y_pred) + Lambda * KL_loss(y_true, y_pred))
        return vae_loss
    
    vae.compile(optimizer='rmsprop', loss=VAE_loss, metrics=[recon_loss, KL_loss])
    #vae.summary()
    #plot_model(vae, to_file='vae_conv_keras_drums_architecture.pdf', show_shapes=True)

    csv_logger = CSVLogger('training_vae_conv_keras_drums.csv')

    if args.weights:
        vae.load_weights(args.weights)
    else:
        # train the autoencoder
        vae.fit(x_train,
                x_train, # unneeded for standard training but seems to be needed when we specify custom loss/metrics
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_test, x_test),
                callbacks=[csv_logger])
        vae.save_weights('vae_conv_keras_drums_epochs.h5')

