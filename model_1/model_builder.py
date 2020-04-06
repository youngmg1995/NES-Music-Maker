# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 13:39:24 2020

@author: Mitchell

Models.py
~~~~~~~~~

"""

# imports
import tensorflow as tf

# Function for building our first model
def model_1_builder(rnn_units,
                    input_dims,
                    emb_output_dims,
                    batch_size,
                    dropout = None):
    '''
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
    P1_X = tf.keras.layers.Embedding(input_dims[0],
                                     emb_output_dims[0],
                                     batch_input_shape=[batch_size, None]
                                     )(P1_input)
    P2_X = tf.keras.layers.Embedding(input_dims[1],
                                     emb_output_dims[1],
                                     batch_input_shape=[batch_size, None]
                                     )(P2_input)
    TR_X = tf.keras.layers.Embedding(input_dims[2],
                                     emb_output_dims[2],
                                     batch_input_shape=[batch_size, None]
                                     )(TR_input)
    NO_X = tf.keras.layers.Embedding(input_dims[3],
                                     emb_output_dims[3],
                                     batch_input_shape=[batch_size, None]
                                     )(NO_input)
    ST_X = tf.keras.layers.Embedding(input_dims[4],
                                     emb_output_dims[4],
                                     batch_input_shape=[batch_size, None]
                                     )(ST_input)
        
    # Layer 3 - LSTM: The bulk of our model an LSTM layer (RNN)
    P1_X = tf.keras.layers.LSTM(rnn_units,
                             return_sequences=True,
                             recurrent_initializer='glorot_uniform',
                             recurrent_activation='sigmoid',
                             stateful=True
                             )(P1_X)
    P2_X = tf.keras.layers.LSTM(rnn_units,
                             return_sequences=True,
                             recurrent_initializer='glorot_uniform',
                             recurrent_activation='sigmoid',
                             stateful=True
                             )(P2_X)
    TR_X = tf.keras.layers.LSTM(rnn_units,
                             return_sequences=True,
                             recurrent_initializer='glorot_uniform',
                             recurrent_activation='sigmoid',
                             stateful=True
                             )(TR_X)
    NO_X = tf.keras.layers.LSTM(rnn_units,
                             return_sequences=True,
                             recurrent_initializer='glorot_uniform',
                             recurrent_activation='sigmoid',
                             stateful=True
                             )(NO_X)
    ST_X = tf.keras.layers.LSTM(rnn_units,
                             return_sequences=True,
                             recurrent_initializer='glorot_uniform',
                             recurrent_activation='sigmoid',
                             stateful=True
                             )(ST_X)
    
    # Optional Layer - Dropout: Ignores set ratio of model features to make
    #   model generalize better through training
    if dropout:
        P1_X = tf.keras.layers.Dropout(dropout)(P1_X)
        P2_X = tf.keras.layers.Dropout(dropout)(P2_X)
        TR_X = tf.keras.layers.Dropout(dropout)(TR_X)
        NO_X = tf.keras.layers.Dropout(dropout)(NO_X)
        ST_X = tf.keras.layers.Dropout(dropout)(ST_X)
    
    # Layer 4 - Dense: Final layer that produces our outputs. In this model
    #   we will concatenate 4 Dense linear layers. Each output is a
    #   distribution over the possible notes for each of the 4 streams in
    #   the 8-bit NES music.
    P1_output = tf.keras.layers.Dense(input_dims[0], name = 'P1')(P1_X)
    P2_output = tf.keras.layers.Dense(input_dims[1], name = 'P2')(P2_X)
    TR_output = tf.keras.layers.Dense(input_dims[2], name = 'TR')(TR_X)
    NO_output = tf.keras.layers.Dense(input_dims[3], name = 'NO')(NO_X)
    ST_output = tf.keras.layers.Dense(input_dims[4], name = 'ST')(ST_X)
    
    # Define final model
    model = tf.keras.Model(inputs  = [P1_input, P2_input,
                                      TR_input, NO_input, ST_input],
                           outputs = [P1_output, P2_output,
                                      TR_output, NO_output, ST_output]
                           )
    
    return model
    
            
            
            