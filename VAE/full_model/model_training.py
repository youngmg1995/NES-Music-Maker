# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 17:14:19 2020

@author: Mitchell

model_training.py
~~~~~~~~~~~~~~~~~
This file serves as a script for building and training our VAE model. To do
so we used the VAE and DataSequence classes defined in the file `VAE.py`, as
well as helper functions from the file `dataset_utils` for loading and parsing
our datasets.

The user has the the ability to specify several parameters that control the
loading of our data, the structure of our model, as well as the traininig plan
for our model. After training is complete the script also plots metrics tracked
during training and saves the final model.

"""

# Imports
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from dataset_utils import load_training, load_validation
from VAE import VAE, DataSequence
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os, time, json


### Load Data
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Parameters for shape of dataset (note these are also used for model def. and
# training.)
measures = 8
measure_len = 96

# training
training_foldername = '../../nesmdb24_seprsco/train/'
train_save_filename = 'transformed_dataset.json'
dataset , labels2int_map , int2labels_map = \
    load_training(training_foldername, train_save_filename,
                  measures = measures, measure_len = measure_len)

# validation
validation_foldername = '../../nesmdb24_seprsco/valid/'
val_save_filename = 'transformed_val_dataset.json'
val_dataset = load_validation(validation_foldername,\
                              labels2int_map, val_save_filename,
                              measures = measures, measure_len = measure_len)


### Build Model
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
### Model Parameters
latent_dim = 124
input_dims = [mapping.shape[0]-1 for mapping in int2labels_map]
dropout = .1
maxnorm = None
vae_b1 , vae_b2 = .02 , .1

# Build Model
model = VAE(latent_dim, input_dims, measures, measure_len, dropout, 
            maxnorm, vae_b1 , vae_b2)
model.build([tf.TensorShape([None, measures, measure_len, input_dims[i]])
             for i in range(4)])
model.summary()


### Train Model
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Training Parameters
batch_size = 100
epochs = 10

# Cost Function
cost_function = model.vae_loss

# Learning_rate schedule
lr_0 = .001
decay_rate = .998
lr_decay = lambda t: lr_0 * decay_rate**t
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lr_decay)

# Optimizer
optimizer = tf.keras.optimizers.Adam()

# Define callbacks
callbacks = [lr_schedule]

# Keras Sequences for Datasets (need to use since one-hot datasets too
# large for storing in memory)
training_seq = DataSequence(dataset, int2labels_map, batch_size)
validation_seq = DataSequence(val_dataset, int2labels_map, batch_size)

# Compile Model
model.compile(optimizer = optimizer,
              loss = cost_function)

# Train model
tic = time.perf_counter()
history = model.fit_generator(generator = training_seq,
                              epochs = epochs)
toc = time.perf_counter()
print(f"Trained Model in {(toc - tic)/60:0.1f} minutes")


### Plot Training Metrics
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
training_loss = history.history['loss']

# Total Loss
plt.figure(1)
plt.plot(training_loss, 'b', label='Training')
plt.title('Loss vs Time')
plt.xlabel('Training Epoch')
plt.ylabel('Avg. Total Loss')
plt.legend()
plt.show()


### Save Model and History
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Save Model Weights
save_model = False
if save_model:
    checkpoint_dir = '.\\training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "model_ckpt")
    model.save_weights(checkpoint_prefix)
    print('Model weights saved to files: '+checkpoint_prefix+'.*')
    
# Save Training History
save_history = False
if save_history:
    checkpoint_dir = '.\\training_checkpoints'
    history_filename = os.path.join(checkpoint_dir, "training_history.json")
    with open(history_filename, 'w') as f:
        json.dump({
            key:[float(value) for value in history.history[key]] 
            for key in history.history
            }, f)
    print('Training history saved to file: '+ history_filename)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#----------------------------------END FILE------------------------------------
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~