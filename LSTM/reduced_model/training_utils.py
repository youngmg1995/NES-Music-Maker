# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 21:29:05 2020

@author: Mitchell

training_utils.py
~~~~~~~~~~~~~~~~~
This file contains functions used for training our NESM models. Functions
contained include the following:
    
    1) train_model
    - Main function that trains our model using the input dataset and
    training parameters. Does so by iteratively calling functions below and
    printing progress along the way.
    
    2) train_step
    - Implements a single training step, evaluating our model against a given
    mini-batch and updating the model weights accordingly.
    
    3) validate_step
    - Computes loss and accuracy of model against batch of validation examples.
    
    4) compute_loss
    - Computes the loss/cost of our model outputs against expected outputs.
    
    5) get_batch
    - Creates mini-batches and expected outputs for said mini-batches that
    can be used for the training and validation steps.
    
"""
### Imports
import tensorflow as tf
import numpy as np
import time


def train_model(model, dataset, val_dataset, 
                seq_length, batch_size, epochs, mini_batches,
                learning_rate, optimizer = tf.keras.optimizers.Adam):
    '''
    Main function used to train our NESM model. Loops for epoch * mini_batches
    # of mini-batches from our training data, during each making a call to the
    train_step function to update our model weights. Throughout this process
    the function prints updates of our training metrics (loss and accuracy
    against training and validation data) and stores these values.

    Parameters
    ----------
    model : tf.keras.Model
        Tensorflow model we are training
    dataset : list of numpy arrays
        Full training dataset. Used to make mini-batches for training.
    val_dataset : list of numpy arrays
        Full validation dataset. Used to make mini-batches for validating
        traininig progress and model performance.
    seq_length : int
        Length of samples used to train our model.
    batch_size : int
        # of samples in each training mini-batch.
    epochs : int
        Number of training epochs
    mini_batches : int
        Number of mini-batches used per epoch
    learning_rate : function
        Function for generating the learning rate input for our optimizer.
        Should be a function of the training epoch.
    optimizer : tf.keras.optimizers.Optimizer, optional
        Optimizer used for updating our model weights. The default is 
        tf.keras.optimizers.Adam.

    Returns
    -------
    loss_history_long : list
        Loss of model output for each mini-batch during training.
    loss_history_short : list
        Loss of model output for last mini-batch at end of each training epoch.
    accuracy_history_long : list
        Accuracy of model output for each mini-batch during training.
    accuracy_history_short : list
        Accuracy of model output for last mini-batch at end of each training
        epoch.
    validation_loss : list
        Loss of model output for validation mini-batch at end of each training
        epoch.
    validation_accuracy : list
        Accuracy of model output for validation mini-batch at end of each
        training epoch.

    '''
    # Initiate metrics for tracking training
    loss_history_long , loss_history_short = [] , []
    accuracy_history_long , accuracy_history_short = [] , []
    validation_loss , validation_accuracy = [] , []
    
    # Train the model
    tic = time.perf_counter()
    for i in range(epochs):
        for j in range(mini_batches):
                    
            # Grab a batch and propagate it through the network
            x_batch, y_batch = get_batch(dataset, seq_length, batch_size)
            loss , accuracy = train_step(x_batch, y_batch, model, compute_loss,
                              optimizer(learning_rate(i)))
            # Update mini-batch metrics
            loss_history_long.append(loss.numpy().mean())  
            accuracy_history_long.append(accuracy.numpy())
        
        # Test against validation data
        x_batch, y_batch = get_batch(val_dataset, seq_length, batch_size)
        val_loss , val_accuracy = validation_step(x_batch, y_batch, model,
                                                  compute_loss)     
        
        # Update epoch metrics
        loss_history_short.append(loss.numpy().mean())
        accuracy_history_short.append(accuracy.numpy())
        validation_loss.append(val_loss.numpy().mean())
        validation_accuracy.append(val_accuracy.numpy())
        
        # Print progress and metrics
        print('Completed Epoch: {}/{}'.format(i+1, epochs))
        print('Training Metrics:      Loss = {:6.4f}    Accuracy = {:.4f}'\
              .format(loss.numpy().mean(), accuracy.numpy()))
        print('Validation Metrics:    Loss = {:6.4f}    Accuracy = {:.4f}'\
              .format(val_loss.numpy().mean(), val_accuracy.numpy()))

    # Print Training Time
    toc = time.perf_counter()
    print(f"Trained Model in {(toc - tic)/60:0.1f} minutes")
    
    return loss_history_long , loss_history_short ,\
           accuracy_history_long , accuracy_history_short ,\
           validation_loss , validation_accuracy


def train_step(x, y, model, loss_function, optimizer):
    '''
    This function performs a single training step against our model. It does
    so by evaluating a single mini-batch example, calculating the loss of our
    model's output, and updating the weights of our model using the chosen
    optimizer and the GradientTape.

    Parameters
    ----------
    x : list of numpy arrays
        Mini-batch example from our trainining data our model will evaluate
        and be updated using.
    y : list of numpy arrays
        Expected or True output for our mini-batch. Loss for the mini-batch
        output will be calculated against these values.
    model : tf.keras.Model
        Th model we are training.
    loss_function : function
        Function for calculating the loss of our model output against the
        expected output.
    optimizer : tf.keras.optimizers.Optimizer
        Optimizer used to update th weights of our model.

    Returns
    -------
    loss : tf.Tensor
        Tensor storing the loss value of our model's output against the
        expected output for the given mini-batch.
    accuracy : tf.Tensor
        Tensor stroring the accuracy value of our model's output against the
        expected output for the given mini-batch.

    '''
    # Use tf.GradientTape()
    with tf.GradientTape() as tape:
          
        # Feed mini_batch into model and find prediction
        y_hat = model(x)
          
        # Find loss for prediction
        loss = loss_function(y, y_hat)
        
        # Find max arguments from predictions
        y_hat = tf.math.argmax(y_hat, axis = -1)
        
        # Compute accuracy of our predictions
        accuracy = tf.metrics.Accuracy()(y, y_hat)
    
    # Now, compute the gradients 
    grads = tape.gradient( loss, model.trainable_variables)
    
    # Apply the gradients to the optimizer so it can update the model accordingly
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss , accuracy


def validation_step(x, y, model, loss_function):
    '''
    Function that tests our model against a mini-batch from our validation
    data. This function is similar to train_step, but it evaluates our
    mini-batch without updating the weights of our model. Instead it is used
    to validate the progress and performance of our model during traininig.

    Parameters
    ----------
    x : list of numpy arrays
        Mini-batch example from our validation data our model will evaluate.
    y : list of numpy arrays
        Expected or True output for our mini-batch. Loss for the mini-batch
        output will be calculated against these values.
    model : tf.keras.Model
        Th model we are testing/training.
    loss_function : function
        Function for calculating the loss of our model output against the
        expected output.

    Returns
    -------
    loss : tf.Tensor
        Tensor storing the loss value of our model's output against the
        expected output for the given mini-batch.
    accuracy : tf.Tensor
        Tensor stroring the accuracy value of our model's output against the
        expected output for the given mini-batch.

    '''
    # Feed mini_batch into model and find prediction
    y_hat = model(x)
      
    # Find loss for prediction
    loss = loss_function(y, y_hat)
    
    # Find max arguments from predictions
    y_hat = tf.math.argmax(y_hat, axis = -1)
    
    # Compute accuracy of our predictions
    accuracy = tf.metrics.Accuracy()(y, y_hat)
    
    return loss , accuracy


def compute_loss(labels, logits):
    '''
    This is the cost function we will be using to train our model. It is just
    the tf.keras.losses.SparseCategoricalCrossentropy cost function available
    in tensorflow. 

    Parameters
    ----------
    labels : numpy array
        Expected or True output labels for the given example(s).
    logits : tf.Tensor
        Output of our model for given example(s). Contains a distribution over
        all possible labels for each output feature.

    Returns
    -------
    loss : tf.Tensor
        Tensor storing the loss value of our model's output against the
        expected output for the given example(s).

    '''
    
    loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss = loss_function(labels,tf.math.log(logits))
    
    return loss


def get_batch(dataset, seq_length, batch_size):
    '''
    This function generates mini-batches from our dataset. It can generate
    a specified # of examples per batch, each with a specified length.

    Parameters
    ----------
    dataset : numpy array
        The dataset we want to generate mini-batches from.
    seq_length : int
        The length of each example sequence in our mini-batch.
    batch_size : int
        The number of example sequences in each mini-batch.

    Returns
    -------
    x_batch : numpy array
        Mini-batch generated from our dataset that will be evaluated by our
        model either for training or validation purposes.
    y_batch : numpy array
        Expected or True output for our mini-batch. Loss for the mini-batch
        output will be calculated against these values. Has the same exact
        shape as the x_batch.

    '''
    # Length of dataset
    M = dataset.shape[0] - 1
    
    # randomly choose the starting indices for the examples in the training batch
    idx = np.random.choice(M-seq_length, batch_size)
    
    # Using indices, slice dataset into batches of length seq_length      
    input_batch = [dataset[i:i+seq_length] for i in idx]
    output_batch = [dataset[i+1:i+seq_length+1] for i in idx]
      
    # x_batch, y_batch provide the true inputs and targets for network training
    x_batch = np.reshape(input_batch, [batch_size, seq_length])
    y_batch = np.reshape(output_batch, [batch_size, seq_length])
    
    return x_batch, y_batch

