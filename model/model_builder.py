# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 13:39:24 2020

@author: Mitchell

model_builder.py
~~~~~~~~~~~~~~~~
This file contains the function `model_builder` used to construct our
tensorflow model. This model will be trained using our NES music data and
then used to generate our own NES music.
"""

import tensorflow as tf


def model_builder(rnn_units,
                  input_dims,
                  emb_output_dims,
                  batch_size,
                  dropout = None):
    '''
    Function that builds our NESM model. This particular model uses a single
    LSTM layer and 5 Dense output layers, one for each feature of our dataset.
    The 5 input data features are each embedded and concatenated so our LSTM
    layer has a single dense vetor input. Additonally, the LSTM output is
    normalized using BatchNormaliztion and has the option of leveraging
    dropout.

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
    dropout : float, optional
        Ratio of LSTM output activiations to dropout. The default is None.

    Returns
    -------
    model : tf.keras.Model
        Tensorflow model we will train using our NESM data and us to generate
        new NES music tracks.

    '''
    # Layer 1 - Inputs: definining inputs and their shapes
    P1_input = tf.keras.Input(shape=(None,),
                          batch_size = batch_size,
                          name='P1_input')
    P2_input = tf.keras.Input(shape=(None,),
                          batch_size = batch_size,
                          name='P2_input')
    TR_input = tf.keras.Input(shape=(None,),
                          batch_size = batch_size,
                          name='TR_input')
    NO_input = tf.keras.Input(shape=(None,),
                          batch_size = batch_size,
                          name='NO_input')
    ST_input = tf.keras.Input(shape=(None,),
                          batch_size = batch_size,
                          name='ST_input') 
    
    
    # Layer 2 - Embedding: Embedding layer to transform our integer inputs
    #   into dense vectors.
    Pulse_Embedding = tf.keras.layers.Embedding(
                                        max(input_dims[0],input_dims[1]),
                                        emb_output_dims[0],
                                        batch_input_shape=[batch_size, None]
                                                )
    Triangle_Embedding = tf.keras.layers.Embedding(
                                        input_dims[2],
                                        emb_output_dims[1],
                                        batch_input_shape=[batch_size, None]
                                                   )
    Noise_Embedding = tf.keras.layers.Embedding(
                                        input_dims[3],
                                        emb_output_dims[2],
                                        batch_input_shape=[batch_size, None]
                                                )
    On_Off_Embedding = tf.keras.layers.Embedding(
                                        input_dims[4],
                                        emb_output_dims[3],
                                        batch_input_shape=[batch_size, None]
                                                 )
    
    P1_X = Pulse_Embedding(P1_input)
    P2_X = Pulse_Embedding(P2_input)
    TR_X = Triangle_Embedding(TR_input)
    NO_X = Noise_Embedding(NO_input)
    OO_X = On_Off_Embedding(ST_input)
    
    # Layer 3 - Input Concatenate: Combine our embedded inputs so we can feed
    #   them into a single LSTM layer
    X = tf.keras.layers.Concatenate()([P1_X, P2_X, TR_X, NO_X, OO_X])
    
    
    # Layer 4 - LSTM: The bulk of our model an LSTM layer (RNN)
    X = tf.keras.layers.LSTM(rnn_units,
                             return_sequences=True,
                             recurrent_initializer='glorot_uniform',
                             recurrent_activation='sigmoid',
                             stateful=True
                             )(X)
    
    # Layer 5 - BatchNormalization: Normalizes our activations across the
    #   mini-batch. This helps increase the learning speed.
    X = tf.keras.layers.BatchNormalization()(X)
    
    # Optional Layer - Dropout: Ignores output activations of LSTM layer
    #   according to given probability.
    if dropout:
        X = tf.keras.layers.Dropout(dropout)(X)
    
    # Layer 6 - Dense: Final layer that produces our outputs. In this model
    #   we will concatenate 5 Dense linear layers. Each output is a
    #   distribution over the possible labels for each input feature.
    P1_output = tf.keras.layers.Dense(input_dims[0], activation = 'softmax',
                                      name = 'P1')(X)
    P2_output = tf.keras.layers.Dense(input_dims[1], activation = 'softmax',
                                       name = 'P2')(X)
    TR_output = tf.keras.layers.Dense(input_dims[2], activation = 'softmax',
                                       name = 'TR')(X)
    NO_output = tf.keras.layers.Dense(input_dims[3], activation = 'softmax',
                                       name = 'NO')(X)
    ST_output = tf.keras.layers.Dense(input_dims[4], activation = 'softmax',
                                       name = 'ST')(X)
    
    # Define final model
    model = tf.keras.Model(inputs  = [P1_input, P2_input,
                                      TR_input, NO_input, ST_input],
                           outputs = [P1_output, P2_output,
                                      TR_output, NO_output, ST_output]
                           )
    
    return model
    
            
            
            