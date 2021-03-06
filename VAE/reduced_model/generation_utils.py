# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 21:29:08 2020

@author: Mitchell

generation_utils.py
~~~~~~~~~~~~~~~~~~~
This file contains helper functions used to generate original NES music
using our trained VAE model. Primarily it contains the following functions:
    
    1) generate_seprsco:
    - Parses outputs generated using our VAE model into valid seprsco files
    for NESM soundtracks.
    
    2) latent_SVD:
    - Function used for effectively sampling from our latent space. Uses
    SVD of latent vectors for training examples to sample from orthogonal
    components of distributions over each latent variable.
    
    3) get_latent_vecs:
    - Used to generate latent vectors for our dataset in smaller batches.
    This is useful when dealing with large datasets.
    
    4) filter_tracks:
    - Filters the outputs of our VAE model so no more than a single note is
    chosen per timestep.
    
    5) plot_track:
    - Plots piano roll representation of the given track.
    
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def generate_seprsco(tracks, int2labels_map):
    '''
    This function is used to parse the outputs of our VAE model into valid
    seprsco data formats. This format can easily be converted into a format for
    playing on the NES or to WAV audio for listening to.

    Parameters
    ----------
    tracks : numpy array or tensorflow tensor
        The output sequence(s) generated by our reduced VAE.
    int2labels_map : numpy array
        This is the inverse mapping used to de-vectorize our sequence.
        More simply, it maps our sequence back to the original label values 
        from the integer representations.

    Returns
    -------
    seprsco_tracks : list of tuples
        List containing each valid seprsco track extracted from our input
        sequence.

    '''
    # Add 0 values to tracks (the placeholder for no note at each timestep)
    zero_notes_shape = tracks.shape[:-1] +[1]
    zero_notes = tf.zeros(zero_notes_shape)
    tracks = tf.concat([zero_notes, tracks], axis=-1)
    
    # Set 0 values to 1 where we have no note (convert to numpy for this)
    tracks = tracks.numpy()
    time_step_sums = tracks.sum(axis=-1)
    zero_note_indxs_1 = np.where(time_step_sums == 0)
    zero_note_indxs_2 = list(zero_note_indxs_1)
    zero_note_indxs_2.append(np.zeros(zero_note_indxs_1[0].shape, dtype='int64'))
    zero_note_indxs_2 = tuple(zero_note_indxs_2)
    tracks[zero_note_indxs_2] = 1
    
    # Grab argmax as note value at each timestep
    tracks = np.argmax(tracks, axis = -1)
    
    # Grab # of tracks
    n = tracks.shape[0]
    
    # Flatten each track to single array
    flat_tracks = tracks.reshape((n,-1))
    
    # Devectorize track using int2labels_map
    flat_tracks = int2labels_map[flat_tracks]
    
    # Initialize array to store seprsco formatted tracks
    seprsco_tracks = []
    
    # Iterate over tracks to reformat as seprsco
    for track in flat_tracks:
        score = np.zeros((track.shape[0],4), np.uint8)
        score[:,0] = track
        nsamps = 1839 * score.shape[0]
        rate = 24.0
        seprsco_tracks.append((rate, nsamps, score))
        
    return seprsco_tracks


def latent_SVD(latent_vecs, rand_vecs, plot_eigenvalues = True):
    '''
    Function used for effectively sampling from our latent variable space.
    Uses singular value decomposition, SVD, of the latent vectors for our
    training dataset to sample from the orthogonal components of the
    distribution over our latent variables.

    Parameters
    ----------
    latent_vecs : numpy array or tensorflow Tensor
        Compressed latent valaues for each sample in our training dataset,
        or whatever dataset we want to model our generated outputs from.
    rand_vecs : numpy array or tensorflow Tensor
        Random samples drawn from normal distribution use to generate latent
        samples.
    plot_eigenvalues : Boolean, optional
        Boolean passed to plot eigenvalues of SVD. The default is True.

    Returns
    -------
    sample_vecs : numpy array
        Sample vectors from latent space.

    '''
    # Perform SVD
    latent_means = np.mean(latent_vecs, axis=0)
    latent_stds = np.std(latent_vecs, axis=0)
    latent_cov = np.cov((latent_vecs - latent_means).numpy(),rowvar=False)
    u, s, v = np.linalg.svd(latent_cov)
    e = np.sqrt(s)
    
    # Plot Eignevalues
    if plot_eigenvalues:
        plt.figure()
        plt.bar(range(1,len(e)+1), e)
        plt.xlabel('Latent Space Component')
        plt.ylabel('Eigenvalue')
        plt.title('SVD of Latent Space')
        plt.show()
    
    # Generate samples from latent space
    sample_vecs = latent_means + np.dot(rand_vecs * e, v)
    
    return np.float32(sample_vecs)
    
def get_latent_vecs(model, dataset, batch_size):
    '''
    Function for grabbing latent vectors for each example in our dataset.
    Typically could just run the entire dataset trhough the model at once, but
    this function can be used to do so in batches when the dataset is too
    large for a single pass-through.
    '''
    # Initialize array for storing batches of latent vecs
    latent_vecs = []
    
    # Find indices of dataset for each batch
    batch_indxs = np.arange(0,dataset.shape[0], batch_size)
    if batch_indxs[-1] != dataset.shape[0]:
        batch_indxs = np.concatenate((batch_indxs,np.array([dataset.shape[0]])))
    
    # Iterate over indices to grab latent vecs for each batch
    for i in range(len(batch_indxs)-1):
        m, n = batch_indxs[i] , batch_indxs[i+1]
        batch = dataset[m:n]
        latent_vecs.append(model.reparameterize(model.encode(batch)))
        
    # Combine batches into single tensor
    latent_vecs = tf.concat(latent_vecs, axis = 0)
    
    return latent_vecs

def filter_tracks(decoded_tracks, p_min):
    '''
    Function used for filtering the outputed tracks from our VAE models.
    Through this filtering process we ensure only notes with a probability
    greater than p_min or chosen for the track and that no more than a single
    note is played at each timestep.
    '''
    # Initialize list for storing filtered tracks
    filtered_tracks = []
    
    # Iterate over each track
    for i in range(decoded_tracks.shape[0]):
        track = decoded_tracks[i]
        
        # Set all but max value at each timestep to 0
        max_indxs = tf.math.argmax(track, axis=-1)
        one_hot_max = tf.one_hot(max_indxs, track.shape[-1], dtype='int64')
        non_max_indxs = np.where(one_hot_max.numpy() == 0)
        filtered_track = track.numpy()
        filtered_track[non_max_indxs] = 0
        
        # Set all values below p_min to 0
        low_values = np.where(filtered_track < p_min)
        filtered_track[low_values] = 0
        
        # Set all values at or above p_min to 1
        high_values = np.where(filtered_track >= p_min)
        filtered_track[high_values] = 1
        
        # Append to list
        filtered_tracks.append(filtered_track)
    
    # Convert numpy aray back to tensor
    filtered_tracks = tf.convert_to_tensor(filtered_tracks)
    
    return filtered_tracks
        
    
def plot_track(decoded_track):
    '''
    Function for plotting piano roll of the given NES track.
    '''
    # Grab # of measures in track
    measures = decoded_track.shape[0]
    
    # Build image of piano roll 2x4 measures
    track = [decoded_track[i] for i in range(measures)]
    image = tf.concat(track, axis = 0)
    
    # Plot image
    plt.figure()
    plt.imshow(np.transpose(image.numpy()),origin = 'lower')
    plt.show()
