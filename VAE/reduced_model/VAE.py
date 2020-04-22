# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 22:30:50 2020

@author: Mitchell

VAE.py
~~~~~~
This file contains code for creating the tensorflow.keras.Model subclass we
will be using for analysing and generating NESM soundtracks. More specifically
this model will be a our reduced VAE or Variational AutoEncoder model. 
This type of model compresses our inputs down to a reduced latent space before
decompressing this data to try and reproduce our initial input. Once 
effectively trained to compress and reproduce our training dataset, we can 
generate new samples by sampling from the reduced latent space and 
decompressing these datapoints.

NOTE: This model only analyses and generates the first melodic voice for our
NESM soundtracks. As such, it only has 1 input and output (compared to the
full model which takes in all 4 feature inputs and outputs each).

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
    Lambda, BatchNormalization, Activation, Dropout
    

class VAE(tf.keras.Model):
    def __init__(self,
                 latent_dim,
                 input_dim,
                 measures,
                 measure_len,
                 dropout = 0.0,
                 maxnorm = None,
                 vae_b1 = 0.02,
                 vae_b2 = 0.1):
        '''
        Initiates a new instance of our VAE model for analysing and generating
        the first meoldic voice of NESM soundtracks.

        Parameters
        ----------
        latent_dim : int
            Dimensionality of our compressed latent space.
        input_dim : list of ints
            Dimensionality or number of unique values for input. Basically the
            number of unique notes in first melodic voice.
        embed_dims: list of ints
            Dimensionlity of embedding for input feature.
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
        self.input_dim = input_dim
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
        
        x_in = Input(shape=(measures,measure_len, input_dim))
        
        x = Reshape((measures, measure_len*input_dim))(x_in)
        
        x = TimeDistributed(Dense(2000, activation = 'relu',
                                  kernel_constraint = kernel_constraint
                                  ))(x)
        x = TimeDistributed(Dense(200, activation = 'relu',
                                  kernel_constraint = kernel_constraint
                                  ))(x)
        
        x = Flatten()(x)
        
        x = Dense(1600, activation = 'relu',
                  kernel_constraint = kernel_constraint
                  )(x)
        
        z_mean = Dense(latent_dim)(x)
        z_log_sigma_sq = Dense(latent_dim)(x)
                
        self.encoder = tf.keras.Model(inputs = x_in,
                                      outputs = [z_mean, z_log_sigma_sq])
        
        # Define Reparameterization
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        z_mean = Input(shape=(latent_dim,))
        z_log_sigma_sq = Input(shape=(latent_dim,))
        
        z = Lambda(self.vae_sampling)([z_mean, z_log_sigma_sq])
        
        self.reparameterize = tf.keras.Model(inputs = [z_mean, z_log_sigma_sq],
                                             outputs = z)
        

        # Define Decoder
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        z = Input(shape=(latent_dim,))
                
        x = Dense(1600, kernel_constraint = kernel_constraint)(z)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        if dropout > 0:
            x = Dropout(dropout)(x)
        
        x = Reshape((measures,200))(x)
        
        x = TimeDistributed(Dense(200, kernel_constraint = kernel_constraint))(x)
        x = TimeDistributed(BatchNormalization())(x)
        x = Activation('relu')(x)
        if dropout > 0:
            x = Dropout(dropout)(x)
        
        x = TimeDistributed(Dense(2000, kernel_constraint = kernel_constraint))(x)
        x = TimeDistributed(BatchNormalization())(x)
        x = Activation('relu')(x)
        if dropout > 0:
            x = Dropout(dropout)(x)
        
        x = TimeDistributed(Dense(measure_len*input_dim, activation = 'sigmoid'))(x)
        
        y = Reshape((measures, measure_len, input_dim))(x)
        
        self.decoder = tf.keras.Model(inputs = z, outputs = y)
        
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
        z_mean , z_log_sigma = self.encoder(x)
        return z_mean , z_log_sigma
      
    def reparameterize(self, z_mean, z_log_sigma):
        '''
        Reparametrizes latent variables by sampling from normal distribution
        with given mean and standard deviatoion.
        '''
        z = self.reparameterize([z_mean , z_log_sigma])
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
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=tf.shape(z_mean), mean=0.,
                                  stddev = self.vae_b1)
        return z_mean + K.exp(z_log_sigma*.5) * epsilon
    
    def vae_loss(self, x, y):
        '''
        Computes the loss between inputs and reproduced outputs of
        our VAE model.
        '''
        z_mean , z_log_sigma_sq = self.z
        xent_loss = tf.keras.losses.MeanSquaredError()(x, y)
        kl_loss = - self.vae_b2 * K.mean(1 + z_log_sigma_sq - K.square(z_mean)\
                                         - K.exp(z_log_sigma_sq), axis=None)
        return xent_loss + kl_loss
