# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 22:30:50 2020

@author: Mitchell

VAE.py
~~~~~~
This file contains code for creating the tensorflow.keras.Model subclass we
will be using for analysing and generating NESM soundtracks. More specifically
this model will be a VAE or Variational AutoEncoder model. This type of model
compresses our inputs down to a reduced latent space before decompressing this
data to try and reproduce our initial input. Once effectively trained to
compress and reproduce our training dataset, we can generate new samples
by sampling from the reduced latent space and decompressing these datapoints.

This particular VAE is slightly unique in that it doesn't exactly attempt to
reproduce the inputs. Instead it returns a value for each possible label for
each output. In other words it provides a probability for each possible output.
When generating our songs we pick the value at each timestep for each feature
with the highest probability.

Along with the Model subclass itself, we have defined several methods used for
training the model within the model. The purpose of all of these methods is to
allow us to train the model using the inherited 'fit' method implicit to all
tensorflow.keras.Models. The methods contained include the following:
    
    1) call
    - Used for evaluating samples or sample-batches using the full VAE model.
    
    2) encode
    - Used for evaluating only the first part of our model, the encoding
    section which compresses our input down to a latent dimension. Note that
    this part of our VAE model is itself a tensorflow.keras.Model.
    
    3) reparameterize
    - Used for evaluating only the second part of our model, the
    reparameterization of our latent dimension. This samples our new latent
    dimension values from a normal distr. with mean and standard deviation
    being the outputs of our encoder. Note that this part of our VAE model is
    itself a tensorflow.keras.Model.
    
    4) decode
    - Used for evaluting just the last part of our model, the decoder section
    that decompresses our latent variables to reproduce the input. Note that
    this part of our VAE model is itself a tensorflow.keras.Model.
    
    5) vae_sampling
    - Function that does sampling for our reparameterize model section.
    
    6) vae_loss
    - Function for calculating the loss between our input and outputs.
    
"""

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Input
from tensorflow.keras.layers import Reshape, TimeDistributed, Dense, Flatten,\
    Lambda, BatchNormalization, Activation, Dropout, Concatenate
from tensorflow.keras.utils import Sequence
import numpy as np

class VAE(tf.keras.Model):
    def __init__(self,
                 latent_dim,
                 input_dims,
                 measures,
                 measure_len,
                 dropout = 0.0,
                 maxnorm = None,
                 vae_b1 = 0.02,
                 vae_b2 = 0.1):
        '''
        Initiates a new instance of our VAE model for analysing and generating
        NESM soundtracks.

        Parameters
        ----------
        latent_dim : int
            Dimensionality of our compressed latent space.
        input_dims : list of ints
            Dimensionality or number of unique values for each input feature.
        embed_dims : list of ints
            Dimensionlity of embedding for each input feature. Note same
            dimensionality is used for first two input features P1 and P2 since
            they are similar melodic voices in our NESM soundtracks.
        measures : int
            Outermost dimension of each sample. # of measures in each sample
            track.
        measure_len : int
            Innermost dimension of each sample. Length of each measure in our
            sample tracks.
        dropout : float, optional
            Dropout rate for each layer. This percent of layer output
            activations get ignored by next layer. Helps imporove fully trained
            performance and reduce overfitting. The default is 0.0.
        maxnorm : float, optional
            Used for putting maxnorm regularization weight contraint on our 
            layer kernels. This limits the size of weights in our model to
            reduce overfitting. Shown to be especially effective in combination
            with dropout. The default is None.
        vae_b1 : float, optional
            Standard deviation for vae_sampling. The default is 0.02.
        vae_b2 : float, optional
            Weight applied to VAE contribution in loss. The default is 0.1.

        Returns
        -------
        None.

        '''
        # Call to Super
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        super(VAE, self).__init__()
        
        # Save Model Parameters
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.latent_dim = latent_dim
        self.input_dims = input_dims
        self.measures = measures
        self.measure_len = measure_len
        self.dropout = dropout
        self.vae_b1 = vae_b1
        self.vae_b2 = vae_b2
        self.maxnrom = maxnorm
        
        # Define Encoder
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if maxnorm:
            kernel_constraint = tf.keras.constraints.MaxNorm(max_value=4)
        else:
            kernel_constraint = None
        
        P1_in = Input(shape=(measures, measure_len, input_dims[0]))
        P2_in = Input(shape=(measures, measure_len, input_dims[1]))
        TR_in = Input(shape=(measures, measure_len, input_dims[2]))
        NO_in = Input(shape=(measures, measure_len, input_dims[3]))
        
        x = Concatenate()([P1_in, P2_in, TR_in, NO_in])
        
        x = Reshape((measures, measure_len*sum(input_dims)))(x)
        
        x = TimeDistributed(Dense(500, kernel_constraint = kernel_constraint))(x)
        x = TimeDistributed(BatchNormalization())(x)
        x = Activation('relu')(x)
        if dropout > 0:
            x = Dropout(dropout)(x)
            
        x = TimeDistributed(Dense(200, kernel_constraint = kernel_constraint))(x)
        x = TimeDistributed(BatchNormalization())(x)
        x = Activation('relu')(x)
        if dropout > 0:
            x = Dropout(dropout)(x)
        
        x = Flatten()(x)
        
        x = Dense(800, kernel_constraint = kernel_constraint)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        if dropout > 0:
            x = Dropout(dropout)(x)
        
        z_mean = Dense(latent_dim)(x)
        z_log_sigma_sq = Dense(latent_dim)(x)
                
        self.encoder = tf.keras.Model(inputs = [P1_in, P2_in, TR_in, NO_in],
                                      outputs = [z_mean, z_log_sigma_sq],
                                      name = 'Encoder')
        
        # Define Reparameterization
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        z_mean = Input(shape=(latent_dim,))
        z_log_sigma_sq = Input(shape=(latent_dim,))
        
        z = Lambda(self.vae_sampling)([z_mean, z_log_sigma_sq])
        
        self.reparameterize = tf.keras.Model(inputs = [z_mean, z_log_sigma_sq],
                                             outputs = z,
                                             name = 'Reparameterizer')
        

        # Define Decoder
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        z = Input(shape=(latent_dim,))
                
        x = Dense(800, kernel_constraint = kernel_constraint)(z)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        if dropout > 0:
            x = Dropout(dropout)(x)
        
        x = Dense(measures*200, kernel_constraint = kernel_constraint)(x)
        x = Reshape((measures,200))(x)
        x = TimeDistributed(BatchNormalization())(x)
        x = Activation('relu')(x)
        if dropout > 0:
            x = Dropout(dropout)(x)
        
        x = TimeDistributed(Dense(500, kernel_constraint = kernel_constraint))(x)
        x = TimeDistributed(BatchNormalization())(x)
        x = Activation('relu')(x)
        if dropout > 0:
            x = Dropout(dropout)(x)
        
        P1 = TimeDistributed(Dense(measure_len*input_dims[0], activation = 'sigmoid'))(x)
        P2 = TimeDistributed(Dense(measure_len*input_dims[1], activation = 'sigmoid'))(x)
        TR = TimeDistributed(Dense(measure_len*input_dims[2], activation = 'sigmoid'))(x)
        NO = TimeDistributed(Dense(measure_len*input_dims[3], activation = 'sigmoid'))(x)
        
        P1_out = Reshape((measures, measure_len, input_dims[0]))(P1)
        P2_out = Reshape((measures, measure_len, input_dims[1]))(P2)
        TR_out = Reshape((measures, measure_len, input_dims[2]))(TR)
        NO_out = Reshape((measures, measure_len, input_dims[3]))(NO)
        
        self.decoder = tf.keras.Model(inputs = z,
                                      outputs = [P1_out,P2_out,TR_out,NO_out],
                                      name = 'Decoder')
        
    def call(self, inputs):
        '''
        Evaluates input sample(s) against full VAE model.
        '''
        z_mean, z_log_sigma_sq = self.encoder(inputs)
        self.z = z_mean, z_log_sigma_sq
        z = self.reparameterize([z_mean, z_log_sigma_sq])
        y = self.decoder(z)
        return y
    
    def encode(self, x):
        '''
        Encodes input by using compressive stage of VAE model.
        '''
        z_mean , z_log_sigma_sq = self.encoder(x)
        return z_mean , z_log_sigma_sq
      
    def reparameterize(self, z_mean, z_log_sigma_sq):
        '''
        Reparametrizes latent variables by sampling from normal distribution
        with given mean and standard deviatoion.
        '''
        z = self.reparameterize([z_mean , z_log_sigma_sq])
        return z
      
    def decode(self, z):
        '''
        Decodes input latent variables using decompressive stage of VAE model.
        '''
        y = self.decode(z)
        return y
    
    def vae_sampling(self, args):
        '''
        Function for sampling latent variables.
        '''
        z_mean, z_log_sigma_sq = args
        epsilon = K.random_normal(shape=tf.shape(z_mean), mean=0.,
                                  stddev = self.vae_b1)
        return z_mean + K.exp(z_log_sigma_sq*.5) * epsilon
    
    def vae_loss(self, x, y):
        '''
        Computes the loss between inputs and reproduced outputs of
        our VAE model.
        '''
        z_mean , z_log_sigma_sq = self.z
        xent_loss = tf.keras.losses.MeanSquaredError()(x, y)
        kl_loss = - self.vae_b2 * K.mean(1 + z_log_sigma_sq - K.square(z_mean)\
                                         - K.exp(z_log_sigma_sq), axis=None)
        return xent_loss + kl_loss/4
    

class DataSequence(Sequence):

        def __init__(self, dataset, int2labels_maps, batch_size):
            self.dataset = dataset
            self.int2labels_maps = int2labels_maps
            self.batch_size = batch_size
            self.batch_indxs = np.arange(0, dataset[0].shape[0], batch_size)
            if self.batch_indxs[-1] != dataset[0].shape[0]:
                self.batch_indxs = np.concatenate((
                    self.batch_indxs, np.array([self.dataset[0].shape[0]])
                    ))

        def __len__(self):
            return tf.math.ceil(self.dataset[0].shape[0] / self.batch_size)

        def __getitem__(self, idx):
            n = len(self.dataset)
            batch_x = [self.dataset[i][self.batch_indxs[i]:self.batch_indxs[i+1]]
                       for i in range(n)]
            batch_x = [tf.one_hot(batch_x[i], self.int2labels_maps[i].shape[0])
                       for i in range(n)]
            batch_x = [batch_x[i][:,:,:,1:] for i in range(n)]

            return batch_x, batch_x
