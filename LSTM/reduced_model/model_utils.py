# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 13:39:24 2020

@author: Mitchell

model_utils.py
~~~~~~~~~~~~~~
This file contains the function `model_builder` used to construct our
tensorflow model. This model will be trained using our NES music data and
then used to generate our own NES music.
"""

import tensorflow as tf


def model_builder(rnn_units,
                  input_dim,
                  emb_output_dim,
                  batch_size,
                  lstm_maxnorm = None,
                  dense_maxnorm = None,
                  lstm_dropout = 0.0,
                  dense_dropout = 0.0):
    '''
    Function that builds our downsized NESM model. This particular model uses
    only the first melodic voice from our NES music tracks. The model has just
    one LSTM layer and one Dense output layer for our singular melodic voice.
    The input data is embedded and concatenated so our LSTM layer has a dense
    vetor input. Additonally, the LSTM output is normalized using
    BatchNormaliztion and has the option of leveraging dropout and max-norm
    regularization.

    Parameters
    ----------
    rnn_units : int
        Number of units for our LSTM layer
    input_dims : list of ints (length 5)
        Dimensionality of each input feature in our dataset. More simply, the
        number of unique values for each input.
    emb_output_dims : list of ints (length 4)
        Dimensionality of the output vectors for each of our embedding layers.
    batch_size : int
        # of samples in each min-batch input to our dataset. Ususally not
        necessary to specify, but is needed for this model since we keep the
        sample length arbitrary.
    lstm_maxnorm : float, optional
        Constraint on the sizes of our LSTM kernel weights. Helps regularize
        the model and prevent overfitting. The default is None.
    dens_maxnorm : float, optional
        Constraint on the sizes of our Dense kernel weights. Helps regularize
        the model and prevent overfitting. The default is None.
    lstm_dropout : float, optional
        Ratio of LSTM unit activiations to dropout. Helps regularize the model,
        prevent overfitting, and imporve performance. The default is 0.
    dense_dropout : float, optional
        Ratio of Dense unit activiations to dropout. Helps regularize the model,
        prevent overfitting, and imporve performance. The default is 0.

    Returns
    -------
    model : tf.keras.Model
        Tensorflow model we will train using our reduced NESM data and use to
        generate new NES music tracks with a single melodic voice.

    '''
    # Layer 1 - Inputs: definining inputs and their shapes
    P1_input = tf.keras.Input(shape=(None,),
                          batch_size = batch_size,
                          name='P1_input')
    
    
    # Layer 2 - Embedding: Embedding layer to transform our integer inputs
    #   into dense vectors.
    P1_X = tf.keras.layers.Embedding(input_dim,
                                     emb_output_dim,
                                     batch_input_shape=[batch_size, None]
                                     )(P1_input)
    
    
    # Layer 3 - LSTM: The bulk of our model an LSTM layer (RNN)
    if lstm_maxnorm:
        kernel_constraint = tf.keras.constraints.MaxNorm(max_value=lstm_maxnorm)
    else:
        kernel_constraint = None        
    
    X = tf.keras.layers.LSTM(rnn_units,
                             return_sequences=True,
                             recurrent_initializer='glorot_uniform',
                             recurrent_activation='sigmoid',
                             kernel_constraint = kernel_constraint,
                             stateful=True,
                             dropout = lstm_dropout
                             )(P1_X)
    
    # Layer 4 - BatchNormalization: Normalizes our activations across the
    #   mini-batch. This helps increase the learning speed.
    X = tf.keras.layers.BatchNormalization()(X)
    
    # Layer 5 - Dropout: Ignores output activations of LSTM layer
    #   according to given probability.
    X = tf.keras.layers.Dropout(dense_dropout)(X)
    
    # Layer 6 - Dense: Final layer that produces our outputs. In this model
    #   we will concatenate 5 Dense linear layers. Each output is a
    #   distribution over the possible labels for each input feature.
    if dense_maxnorm:
        kernel_constraint = tf.keras.constraints.MaxNorm(max_value=dense_maxnorm)
    else:
        kernel_constraint = None
    
    P1_output = tf.keras.layers.Dense(input_dim, activation = 'softmax',
                                      kernel_constraint = kernel_constraint,
                                      name = 'P1')(X)
    
    # Define final model
    model = tf.keras.Model(inputs  = P1_input, outputs = P1_output)
    
    return model
    
            
            
            