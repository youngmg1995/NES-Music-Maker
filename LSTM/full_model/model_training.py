# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 17:14:19 2020

@author: Mitchell

model_training.py
~~~~~~~~~~~~~~~~~
This file serves as a script for building and training our model. To do so we
use helper functions from the files `dataset_utils`, `model_utils`, and
`training_utils`, which respectively help us to load our data, build our
model, and train our model. The user has the ability to specify several
parameters at each of these stages to control the data loaded, the structure
of the model, and the training prescription. After training is complete, the
script also plots graphs describing the training performance and at the end
gives the user the option of saving the model.
"""

# Imports
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from dataset_utils import load_training, load_validation
from model_utils import model_builder
from training_utils import train_model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os


### Load Data
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# training
training_foldername = '../../nesmdb24_seprsco/train/'
train_save_filename = 'transformed_dataset.json'
dataset , labels2int_maps , int2labels_maps = \
    load_training(training_foldername, train_save_filename)

# validation
validation_foldername = '../../nesmdb24_seprsco/valid/'
val_save_filename = 'transformed_val_dataset.json'
val_dataset = load_validation(validation_foldername,\
                              labels2int_maps, val_save_filename)


### Build Model
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Model Parameters
rnn_units = 1024
input_dims = [len(v) for v in int2labels_maps]
emb_output_dims = [128, 128, 32, 16]
batch_size = 100
lstm_maxnorm , dense_maxnorm = None , 4
lstm_dropout , dense_dropout = .5 , .5

# Build Model
model = model_builder(rnn_units, input_dims, emb_output_dims, batch_size,
                      lstm_maxnorm, dense_maxnorm, lstm_dropout, dense_dropout)


# For loading saved model weights
'''
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "model_ckpt")
model.load_weights(checkpoint_prefix)
model.build(tf.TensorShape([batch_size, None, 5]))
'''


### Train Model
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Training Parameters
seq_length = 100
epochs = 50
mini_batches = 100
learning_rate_0 , learning_rate_f= .01, .005
decay_rate = .93#(learning_rate_f/learning_rate_0)**(1/(epochs-1))
learning_rate = lambda t: learning_rate_0 * decay_rate**t
optimizer = tf.keras.optimizers.Adam

# For intermediate model saves
checkpoint_dir = '.\\training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "model_ckpt")
save_step = 10

# Train Model
loss_history_long , loss_history_short , accuracy_history_long ,\
accuracy_history_short , validation_loss , validation_accuracy \
    =\
train_model(model, dataset, val_dataset, seq_length, batch_size, epochs, 
            mini_batches, learning_rate, optimizer,
            checkpoint_prefix, save_step=10)


### Plot Training Metrics
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
plt.figure(1)
plt.plot(range(1,len(loss_history_short)+1), loss_history_short, 'b', label='Training')
plt.plot(range(1,len(validation_loss)+1), validation_loss, 'r', label='Validation')
plt.title('Model: Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.figure(2)
plt.plot(range(1,len(accuracy_history_short)+1), accuracy_history_short, 'b', label='Training')
plt.plot(range(1,len(validation_accuracy)+1), validation_accuracy, 'r', label='Validation')
plt.title('Model: Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.figure(3)
plt.plot(range(1,len(loss_history_long)+1), loss_history_long, 'g')
plt.title('Model: Loss Over Mini-Batches')
plt.xlabel('Mini-Batch')
plt.ylabel('Loss')
plt.show()
plt.figure(4)
plt.plot(range(1,len(accuracy_history_long)+1), accuracy_history_long, 'g')
plt.title('Model: Accuracy Over Mini-Batches')
plt.xlabel('Mini-Batch')
plt.ylabel('Accuracy')
plt.show()


### Save Model
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
save_weights = True
if save_weights:
    checkpoint_dir = '.\\training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "model_ckpt")
    model.save_weights(checkpoint_prefix)
    print('Model weights saved to files: '+checkpoint_prefix+'.*')


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#----------------------------------END FILE------------------------------------
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~