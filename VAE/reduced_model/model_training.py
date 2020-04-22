# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 17:14:19 2020

@author: Mitchell

model_training.py
~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~
This file serves as a script for building and training our reduced VAE model.
To do so we used the VAE class defined in the file `VAE.py`, as well as helper
functions from the file `dataset_utils` for loading and parsing our datasets.

The user has the the ability to specify several parameters that control the
loading of our data, the structure of our model, as well as the traininig plan
for our model. After training is complete the script also plots metrics tracked
during training and saves the final model.

"""

# Imports
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from dataset_utils import load_training, load_validation
from VAE import VAE
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os


### Load Data
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Parameters for shape of dataset (note these are also used for model def.)
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
input_dim = len(int2labels_map) - 1
dropout = .1
maxnorm = None
vae_b1 , vae_b2 = .01, .5#.02 , .1

# Build Model
model = VAE(latent_dim, input_dim, measures, measure_len, dropout, 
            maxnorm, vae_b1 , vae_b2)
model.build(tf.TensorShape([None, measures, measure_len, input_dim]))
model.summary()


### Train Model
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Training Parameters
batch_size = 300
epochs = 10

# Cost Function
cost_function = model.vae_loss

# Optimizer and learning_rate schedule
lr_0 = .001
decay_rate = .998
lr_decay = lambda t: lr_0 #* decay_rate**t
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lr_decay)
optimizer = tf.keras.optimizers.Adam()

# Compile Model
model.compile(optimizer = optimizer,
              loss = cost_function,
              metrics = ['accuracy'])

# Train model
history = model.fit(dataset, dataset,
                    batch_size = batch_size,
                    epochs = epochs,
                    callbacks = [lr_schedule],
                    validation_data = (val_dataset, val_dataset))



### Plot Training Metrics
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
training_loss = history.history['loss']
traininig_accuracy = history.history['accuracy']
val_loss = history.history['val_loss']
val_accuracy = history.history['val_accuracy']


plt.figure(1)
plt.plot(training_loss, 'b', label='Training')
plt.plot(val_loss, 'r', label='Validation')
plt.title('Model: Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.figure(2)
plt.plot(traininig_accuracy, 'b', label='Training')
plt.plot(val_accuracy, 'r', label='Validation')
plt.title('Model: Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


### Save Model
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
save_weights = False
if save_weights:
    checkpoint_dir = '.\\training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "model_ckpt")
    model.save_weights(checkpoint_prefix)
    print('Model weights saved to files: '+checkpoint_prefix+'.*')


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#----------------------------------END FILE------------------------------------
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~