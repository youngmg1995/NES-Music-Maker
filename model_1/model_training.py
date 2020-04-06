# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 17:14:19 2020

@author: Mitchell
"""

# imports
from model_builder import model_1_builder
from nesm_loader import load_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os, time

### load music data
data_foldername = '../nesmdb24_seprsco/train/'
save_filename = 'transformed_dataset.json'
dataset , labels2int_maps , int2labels_maps = load_data(data_foldername, save_filename)
    
### defining functions to build batches for training
def get_batch(dataset, seq_length, batch_size):
    '''
    Function used to create mini-batches for training our OrchestralModels.
    '''
    # Length of dataset
    M = dataset[0].shape[0] - 1
    
    # randomly choose the starting indices for the examples in the training batch
    idx = np.random.choice(M-seq_length, batch_size)
    
    # iterate over each feature to grab batches
    x_batch , y_batch = [] , []
    for data in dataset:
          
        input_batch = [data[i:i+seq_length] for i in idx]
        output_batch = [data[i+1:i+seq_length+1] for i in idx]
          
        # x_batch, y_batch provide the true inputs and targets for network training
        x_batch.append(np.reshape(input_batch, [batch_size, seq_length]))
        y_batch.append(np.reshape(output_batch, [batch_size, seq_length]))
    
    return x_batch, y_batch

### Defining our cost function. Needs to handle all 4 model outputs.
def compute_loss(labels, logits, weights=None):
    '''
    '''
    if weights==None:
        weights = []
        for i in range(len(labels)):
            weights.append(1.)
    loss = 0.
    weight_sum = 0.
    loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    for lb, lg, wt in zip(labels, logits, weights):
        loss += wt * loss_function(lb,lg)
                                                              
        weight_sum += wt
    loss /= weight_sum
    
    return loss

### Defining function for a single training step for our model. 
# This function calculates the loss for that mini-batch and updates the weights
# of our model.
def train_step(x, y, model, loss_function, optimizer): 
    # Use tf.GradientTape()
    with tf.GradientTape() as tape:
          
        # Feed mini_batch into model and find prediction
        y_hat = model(x)
          
        # Find loss for prediction
        loss = loss_function(y, y_hat)
    
    # Now, compute the gradients 
    grads = tape.gradient( loss, model.trainable_variables)
    
    # Apply the gradients to the optimizer so it can update the model accordingly
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss



### Building and Training Model 1
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
### Define parameters for our models and training
rnn_units = 256
input_dims = [len(v) for v in int2labels_maps]
emb_output_dims = [128, 128, 128, 32, 8]
batch_size = 100
seq_length = 200
learning_rate = .01
rate_decay = .95
epochs = 100
epoch_steps = 100
# Choosing optimizer
optimizer = tf.keras.optimizers.Adam

# Build model
model = model_1_builder(rnn_units, input_dims, emb_output_dims, batch_size)

# Train the model
loss_history_long = []
loss_history_short = []
tic = time.perf_counter()
for i in range(epochs):
    for j in range(epoch_steps):
                
        # Grab a batch and propagate it through the network
        x_batch, y_batch = get_batch(dataset, seq_length, batch_size)
        loss = train_step(x_batch, y_batch, model, compute_loss,
                          optimizer(learning_rate * rate_decay**i))
        # Update loss_history_long
        loss_history_long.append(loss.numpy().mean())        

    # Update loss_history_short
    loss_history_short.append(loss.numpy().mean())
    # Print progress and loss
    print('Completed Epoch: {}/{}    ,    Loss = {:6.4f}'.format(i+1, epochs, loss.numpy().mean()))

# Print Training Time
toc = time.perf_counter()
print(f"Created network1 in {(toc - tic)/60:0.1f} minutes")

#plotting losses
plt.figure(1)
plt.plot(range(1,len(loss_history_short)+1), loss_history_short, 'b')
plt.title('Model 2: Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
plt.figure(2)
plt.plot(range(1,len(loss_history_long)+1), loss_history_long, 'r')
plt.title('Model 2: Loss Over Mini-Batches')
plt.xlabel('Mini-Batch')
plt.ylabel('Loss')
plt.show()


# Save trained model weights
checkpoint_dir = '.\\training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "model_1_ckpt")
model.save_weights(checkpoint_prefix)
print('Model weights saved to files: '+checkpoint_prefix+'.*')


