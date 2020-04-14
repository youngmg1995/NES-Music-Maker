# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 22:08:58 2020

@author: Mitchell

dataset_utils.py
~~~~~~~~~~~~~~~~
This file contains various functions for parsing and loading in our datasets
necessary for training and validating our model. Note that in this model we
only use the first melodic voice from our seprsco files. In doing so, we aim
to train our first model using a simpler dataset. Functions contained include
the following:
    
    1) load_training
    - Used to load and parse our training data to format nescessary for
    training. Does so by callng either `load_seprsco` or `reload_data`.
    
    2) load_seprsco
    - Loads training data from seprsco files and parses to proper format 
    needed for training.
    
    3) reload_data
    - Loads training data from file location containing pre-parsed data
    from previous training session. (No need to redo work)
    
    4) vectorizer
    - Transforms training data to integer representations that are simpler to
    train. Returns trasformed data as well as mappings for transformation and
    inverse mappings.
    
    5) load_validation
    - Loads validation data from seprsco files and parses to proper format.
    
    6) load_song
    - Loads data from a single seprsco file and parses to proper format.
    
"""

import glob, pickle, json
import numpy as np
import os, random


def load_training(training_foldername, save_filename=None):
    '''
    Function that loads training data from prescribed folder or from given 
    savefile location if it already exists. More specifically for this model
    it loads the NES music from seprsco file formats and reshaped/vectorizes
    them to the format necessary for our model.

    Parameters
    ----------
    training_foldername : string
        Directory containing the seprsco files for the training data.
    save_filename : string, optional
        Filename specifying where to load pre-parsed training data from and/or
        location to save files if parsing them for the first time.
        The default is None.

    Returns
    -------
    dataset : numpy array
        Training data for fitting our model. Contains first melodic voice
        from seprsco data for 4,502 NES music tracks. Pre-parsed and vectorized
        to integer representation.
    labels2int_map : dictionary
        Dictionary used to map original training data to integers.
    int2labels_map : numpy array
        Inverse mapping of labels2int_map. Used to map our training data back
        from integers to original format.

    '''
    # Check if save_filename exists. 
    # If so load pre-transormed data from there using 'reload_data'.
    # If not, load orginal seprsco files using 'initial_load_data'.
    if save_filename and os.path.exists(save_filename):
        dataset , labels2int_maps , int2labels_maps = reload_data(save_filename)
    else:
        dataset , labels2int_maps , int2labels_maps = load_seprsco(training_foldername, save_filename)
    
    return dataset , labels2int_maps , int2labels_maps
        
     
def load_seprsco(training_foldername, save_filename=None):
    '''
    Main function used to load and parse training data from seprsco files. Then
    saves the parsed data to save_filename if given.

    Parameters
    ----------
    training_foldername : string
        Directory containing the seprsco files for the training data.
    save_filename : string, optional
        Filename specifying where to save parsed training data.

    Returns
    -------
    dataset : numpy array
        Training data for fitting our model. Contains first melodic voice
        from seprsco data for 4,502 NES music tracks. Pre-parsed and vectorized
        to integer representation.
    labels2int_map : dictionary
        Dictionary used to map original training data to integers.
    int2labels_map : numpy array
        Inverse mapping of labels2int_map. Used to map our training data back
        from integers to original format.

    '''
    ### load all the files
    song_files = glob.glob(training_foldername+'*')
    
    # Shuffle our song files so similar songs aren't all in a row
    random.shuffle(song_files)

    # datapoint used to indicate start/end of each track
    start_end = np.array([109],np.uint8)
    
    # Initiate start of our dataset
    dataset = start_end.copy()
    
    # Parameters used during loop
    counter = 0
    interval = 1
    interval_length = len(song_files) / 10
    
    # Initiate loop to load in data, combining all tracks into one long track
    print('Loading Training Data from Directory: "{}"'.format(training_foldername))
    for file in song_files:
        # Open file
        with open(file, 'rb') as f:
            # Load in data from file. Only seprsco data will be used
            rate, nsamps, score = pickle.load(f)
            
            # Remove first melodic voice from score and reshape
            n = score.shape[0]
            melody = score[:,0].reshape(n)
            
            # Add start/end note to end of track and reformat
            melody = np.concatenate((melody, start_end))
            
            # Append track to dataset
            dataset = np.concatenate((dataset,melody))
            
        # Print update everytime 10% of files loaded
        counter += 1 
        if counter / interval_length >= interval:
            print('Files Loaded: {}%'.format(interval*10))
            interval += 1
            
    # Loading Complete and return datasets
    print('Loading Training Files Complete')
    
    # Vectorize the dataset
    dataset , labels2int_map , int2labels_map = vectorizer(dataset)
    
    # Save transformed dataset and mappings if filename given
    if save_filename:
        print('Saving Transformed Training Data and Mappings to Filename "{}"'.format(save_filename))
        data = {"dataset": dataset.tolist(),
                "unique_values": int2labels_map.tolist()
                }
        with open(save_filename, 'w') as f:
            json.dump(data, f)
        print('Saving Data Complete')

    # Print that we are done
    print('Loading Training Data Complete')
    
    return dataset , labels2int_map , int2labels_map


def reload_data(save_filename, dtype = np.uint8):
    '''
    Function that loads pre-parsed training data from specified save_filename.

    Parameters
    ----------
    save_filename : string
        Filename specifying location of pre-parsed training data.
    dtype : dtype, optional
        Used to specify the dtype of our original training dataset. This may be
        necessary to specify because certain dtypes are not the implicit dtypes
        loaded from our json save_file. The default is np.uint8.

    Returns
    -------
    dataset : numpy array
        Training data for fitting our model. Contains first melodic voice
        from seprsco data for 4,502 NES music tracks. Pre-parsed and vectorized
        to integer representation.
    labels2int_map : dictionary
        Dictionary used to map original training data to integers.
    int2labels_map : numpy array
        Inverse mapping of labels2int_map. Used to map our training data back
        from integers to original format.

    '''
    print('Reloading Pre-Transformed Training Data from Filename "{}"'.format(save_filename))
    
    # Load in data with json
    with open(save_filename, 'r') as f:
        data = json.load(f)
    
    # Build datasets and mappings
    dataset = np.array(data["dataset"])
    
    labels2int_map = {u:i for i, u in enumerate(np.array(data["unique_values"],dtype))}
    
    int2labels_map = np.array(data["unique_values"], dtype)
    # Print that we are done
    print('Loading Training Data Complete')
    
    return dataset , labels2int_map , int2labels_map


def vectorizer(dataset):
    '''
    Functon that transforms our dataset from arbitrary labels to integers. In
    doing so it indentifies the unique lables/values in our dataset. These
    unique values are used to create the mapping fom lables to integers as well
    as the inverse mapping.

    Parameters
    ----------
    dataset : numpy array
        Input dataset is assumed to be a numpy arrays with shape (N,). This
        is format our loaded training seprsco data in the load_training
        function.

    Returns : 
    -------
    vectorized_dataset : numpy array
        Transformed dataset where each datapoint is now represented using an
        integer. Has same shape/structure as originial input dataset.
    
    labels2int_dicts : dictionary
        Transformation dictionary used to vectorize input dataset. Must apply
        dictionary in loop over dataset to transform.
        
    int2labels_dicts : numpy array
        Numpy array that acts as inverse to above dictionary for transforming
        our dataset back from to its original label format. In this format the
        entire integer feature array can be transformed back to its orginal
        form by passing the whole array (no need to loop over data).
    '''
    print('Vectorizing Training Data')
    
    # Find unique labels
    unique_labels = sorted(set(dataset))
    # Create mappings
    labels2int_map = {u:i for i, u in enumerate(unique_labels)}
    int2labels_map = np.array(unique_labels)
    
    # Loop over data and vectorize to ints using mapping
    vectorized_dataset = np.zeros(dataset.shape,np.int32)
    for j in range(dataset.shape[0]):
        vectorized_dataset[j] = labels2int_map[dataset[j]]
    
    print('Vectorizing Complete')
    
    return vectorized_dataset , labels2int_map , int2labels_map


def load_validation(validation_folder, labels2int_map, save_filename=None):
    '''
    Function used to load validation data and parse it into necessary format.
    Also saves parsed validation data to save_filename if given.

    Parameters
    ----------
    validation_folder : string
        Directory containing the seprsco files for our validation dataset.
    labels2int_maps : list of dictionaries
        List containing the transformation mappings used to vectorize the
        training data. These are also used to vectorize the validation data.
    save_filename : string, optional
        Filename specifying where to load pre-parsed validation data from
        and/or where to save the now parsed validation data if loaded for the
        first time. The default is None.

    Returns
    -------
    val_dataset : numpy arrays
        Parsed validation dataset used for training our model. 

    '''
    # Check if savefile exists. 
    # If so load pre-transormed data from there
    # If not, load orginal seprsco files
    if save_filename and os.path.exists(save_filename):
        print('Reloading Pre-Transformed Validation Data from Filename "{}"'.format(save_filename))
        
        # Load in data with json
        with open(save_filename, 'r') as f:
            data = json.load(f)
    
        # Build datasets and mappings
        val_dataset = np.array(data["val_dataset"])
        
    else:
        print('Loading Validation Data from Directory: "{}"'.format(validation_folder))
        
        ### load all the files
        song_files = glob.glob(validation_folder+'*')
        
        # Shuffle our song files so similar songs aren't all in a row
        random.shuffle(song_files)
    
        # datapoint used to indicate start/end of each track
        start_end = np.array([109],np.uint8)
        
        # Initiate start of our dataset
        dataset = start_end.copy()
        
        # Initiate loop to load in data, combining all tracks into one long track
        for file in song_files:
            # Open file
            with open(file, 'rb') as f:
                # Load in data from file. Only seprsco data will be used
                rate, nsamps, score = pickle.load(f)
                
                # Remove first melodic voice from score and reshape
                n = score.shape[0]
                melody = score[:,0].reshape(n)
                
                # Add start/end note to end of track and reformat
                melody = np.concatenate((melody, start_end))
                
                # Append track to dataset
                dataset = np.concatenate((dataset, melody))
        
        # Vectorize the dataset
        val_dataset = np.zeros(dataset.shape,np.int32)
        for j in range(dataset.shape[0]):
            val_dataset[j] = labels2int_map[dataset[j]]
        
        # Save transformed dataset if filename given
        if save_filename:
            print('Saving Transformed Validation Data to Filename "{}"'.format(save_filename))
            data = {"val_dataset": val_dataset.tolist()}
            with open(save_filename, 'w') as f:
                json.dump(data, f)
            print('Saving Data Complete')
    
    # Print that we are done
    print('Loading Validation Data Complete')
    
    return val_dataset
    

def load_track(song_filename, labels2int_map):
    '''
    Function used for loading a single seprsco file. Similar to
    load_validation function, but only loads a single file and has no option
    to save the parsed data.

    Parameters
    ----------
    song_filename : string
        Filename specifying location of seprsco file to load and parse.
    labels2int_maps : dictionary
        The transformation mapping used to vectorize the training data.
        It are also used to vectorize the track.

    Returns
    -------
    track_data : numpy array
        Parsed data for the given track that is ready to feed into our model.

    '''
    # Load in data from file. Only seprsco data will be used
    with open(song_filename, 'rb') as f:
        rate, nsamps, score = pickle.load(f)
    
    # Isolate first melodic voice and reshape
    n = score.shape[0]
    dataset = score[:,0].reshape(n)
    
    # Vectorize the data
    track_data = np.zeros(dataset.shape,np.int32)
    for j in range(dataset.shape[0]):
        track_data[j] = labels2int_map[dataset[j]]
            
    return track_data
    
    