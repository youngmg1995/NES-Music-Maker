# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 22:08:58 2020

@author: Mitchell

dataset_utils.py
~~~~~~~~~~~~~~~~
This file contains various functions for parsing and loading in our datasets.
These datasets will be used for training our VAE model, validatig the model,
and generating new NES soundtracks using our model.

NOTE - For this model we are only analysing and generating the first melodic
voice for NESM soundtracks. As such, all the function here only load, parse,
and save this one feture from our files.

Functions contained include the following:
    
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


def load_training(training_foldername, save_filename=None,
                  measures = 8, measure_len = 96):
    '''
    Function that loads training data from prescribed folder or from given 
    savefile location if it already exists. Dataset returned is in the specific
    format required for training the reduced VAE class models defined in this
    folder. For our reduced model, this means we will only use the first
    melodic voice from each seprsco track.

    Parameters
    ----------
    training_foldername : string
        Directory containing the NESM seprsco files for the training data.
    save_filename : string, optional
        Filename specifying where to load pre-parsed training data from and/or
        location to save files if parsing them for the first time.
        The default is None.
    measures : int
        This is a parameter used for structuring the returned training dataset.
        More specifically, it is the outermost dimension for each training
        sample passed to our VAE model. In more layman's terms, if each sample
        is a song, then this is the number of full measures per song. The 
        default is 8.
    measure_len : int
        This is a parameter used for structuring the returned training dataset.
        More specifically, it is the innermost dimension for each
        training sample passed to our VAE model. In more layman's terms, if
        each sample is a song, then this is the length of or the number of
        notes/samples in each measure. It should be noted that all seprsco
        tracks translate to a tempo of 24 notes/samples per second. The default
        is 96.

    Returns
    -------
    dataset : numpy array
        Training data for fitting our VAE model. The dataset contains 1 numpy
        array representing the first melodic voice from each NESM soundtrack.
        Shape is [# tracks, measures, measure_length].
    labels2int_maps : dictionary
        Dictionary used to map original training data to integers.
    int2labels_maps : numpy array
        Inverse mapping of labels2int_map. Used to map our training data back
        from integers to original format.

    '''
    # Check if save_filename exists. 
    # If so load pre-transormed data from there using 'reload_data'.
    # If not, load orginal seprsco files using 'initial_load_data'.
    if save_filename and os.path.exists(save_filename):
        dataset , labels2int_maps , int2labels_maps = reload_data(save_filename)
    else:
        dataset , labels2int_maps , int2labels_maps = \
           load_seprsco(training_foldername, save_filename, measures, measure_len)
    
    return dataset , labels2int_maps , int2labels_maps
        
     
def load_seprsco(training_foldername, save_filename=None,
                 measures = 8, measure_len = 96):
    '''
    Main function used to load and parse training data from seprsco files. Then
    saves the parsed data to save_filename if given.

    Parameters
    ----------
    training_foldername : string
        Directory containing the NESM seprsco files for the training data.
    save_filename : string, optional
        Filename specifying where to load pre-parsed training data from and/or
        location to save files if parsing them for the first time.
        The default is None.
    measures : int
        This is a parameter used for structuring the returned training dataset.
        More specifically, it is the outermost dimension for each training
        sample passed to our VAE model. In more layman's terms, if each sample
        is a song, then this is the number of full measures per song. The 
        default is 8.
    measure_len : int
        This is a parameter used for structuring the returned training dataset.
        More specifically, it is the innermost dimension for each
        training sample passed to our VAE model. In more layman's terms, if
        each sample is a song, then this is the length of or the number of
        notes/samples in each measure. It should be noted that all seprsco
        tracks translate to a tempo of 24 notes/samples per second. The default
        is 96.

    Returns
    -------
    dataset : numpy array
        Training data for fitting our VAE model. The dataset contains 1 numpy
        array representing the first melodic voice from each NESM soundtrack.
        Shape is [# tracks, measures, measure_length].
    labels2int_maps : dictionary
        Dictionary used to map original training data to integers.
    int2labels_maps : numpy array
        Inverse mapping of labels2int_map. Used to map our training data back
        from integers to original format.

    '''
    # Set sample_length of samples based of measures and measure_len
    sample_length = measures * measure_len
    
    # load all the filenames
    song_files = glob.glob(training_foldername+'*')
    
    # Shuffle our song files so similar songs aren't all in a row
    random.shuffle(song_files)
    
    # Initiate our dataset
    dataset = []
    
    # Parameters used during loop
    counter = 0
    interval = 1
    interval_length = len(song_files) / 10
    
    # Initiate loop to load in data
    print('Loading Training Data from Directory: "{}"'.format(training_foldername))
    for file in song_files:
        # Open file
        with open(file, 'rb') as f:
            # Load in data from file. Only score data will be used
            # Specifically first voice in score (score[:,0])
            rate, nsamps, score = pickle.load(f)
            
            # parse song if length is above sample_length
            n = score.shape[0]
            if n >= sample_length:
                # Remove first melodic voice from score and reshape
                melody = score[:,0].reshape(n)
                
                # Remove first m segments of score that are exactly the
                # length we need (sample_length)
                m = n // sample_length
                for i in range(m):
                    # append segment to dataset
                    dataset.append(melody[sample_length*i:sample_length*(i+1)])
            
        # Print update everytime 10% of files loaded
        counter += 1 
        if counter / interval_length >= interval:
            print('Files Loaded: {}%'.format(interval*10))
            interval += 1
            
    # Loading Complete and return datasets
    print('Loading Training Files Complete')
    
    # Vectorize the dataset
    dataset , labels2int_map , int2labels_map = vectorizer(dataset, measures, measure_len)
    
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
    dtypes : dtypes, optional
        Used to specify dtype of first melodic voice in our original training
        dataset. This may be necessary to specify because certain dtypes are
        not the implicit dtypes loaded from our json save_file. The default is
        set to np.uint8 which is the dtype for all data in our seprsco scores.

    Returns
    -------
    dataset : numpy array
        Training data for fitting our VAE model. The dataset contains 1 numpy
        array representing the first melodic voice from each NESM soundtrack.
        Shape is [# tracks, measures, measure_length].
    labels2int_maps : dictionary
        Dictionary used to map original training data to integers.
    int2labels_maps : numpy array
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


def vectorizer(dataset, measures = 8, measure_len = 96):
    '''
    Functon that transforms our dataset from arbitrary labels to integers. In
    doing so it indentifies the unique lables/values in our dataset. These
    unique values are used to create the mapping fom lables to integers as well
    as the inverse mapping.

    Parameters
    ----------
    dataset : lists of numpy arrays
        Dataset we are vectorizing. Assumed to be list of numpy arrays. 
        Each numpy array represents a separate training sample.
    measures : int
        Outermost dimension used for reshaping our vectorized dataset samples.
        The default is 8.
    measure_len : int
        Innermost dimension used for reshaping our vectorized dataset samples.
        The default is 96.

    Returns : 
    -------
    vectorized_dataset : numpy array
        Transformed dataset where each datapoint is now represented using an
        integer. All tracks are now combined to single array with shape
        [# tracks, measures, measure_length].
    labels2int_maps : dictionary
        Dictionary used to map original training data to integers.
    int2labels_map : numpy array
        Inverse mapping of labels2int_map. Used to map our training data back
        from integers to original format.
        
    '''
    print('Vectorizing Training Data')
    
    # Find unique labels
    unique_labels = sorted(set(np.concatenate(dataset)))
    # Create mappings
    labels2int_map = {u:i for i, u in enumerate(unique_labels)}
    int2labels_map = np.array(unique_labels)
    
    # Loop over data and vectorize to ints using mapping and reshape
    vectorized_dataset = []
    for data in dataset:
        new_data = np.zeros(data.shape, np.int32)
        for i in range(len(data)):
            new_data[i] = labels2int_map[data[i]]
        new_data = new_data.reshape(measures, measure_len)
        vectorized_dataset.append(new_data)
    
    # Convert list to single numpy array
    vectorized_dataset = np.array(vectorized_dataset)
    
    print('Vectorizing Complete')
    
    return vectorized_dataset , labels2int_map , int2labels_map


def load_validation(validation_folder, labels2int_map, save_filename=None,
                    measures = 8, measure_len = 96):
    '''
    Function used to load validation data and parse it into necessary format.
    Also saves parsed validation data to save_filename if given.

    Parameters
    ----------
    validation_folder : string
        Directory containing the seprsco files for our validation dataset.
    labels2int_maps : list of dictionaries
        List containing the transformation mapping used to vectorize the
        training data. This is also used to vectorize the validation data.
    save_filename : string, optional
        Filename specifying where to load pre-parsed validation data from
        and/or where to save the now parsed validation data if loaded for the
        first time. The default is None.

    Returns
    -------
    val_dataset : numpy array
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
    
        # Build dataset
        val_dataset = np.array(data["val_dataset"])
        
    else:
        print('Loading Validation Data from Directory: "{}"'.format(validation_folder))
        
        # Set sample_length of samples based of measures and measure_len
        sample_length = measures * measure_len
        
        # load all the filenames
        song_files = glob.glob(validation_folder+'*')
        
        # Shuffle our song files so similar songs aren't all in a row
        random.shuffle(song_files)
        
        # Initiate start of our dataset
        dataset = []
        
        # Initiate loop to load in data
        for file in song_files:
            # Open file
            with open(file, 'rb') as f:
                # Load in data from file. Only seprsco data will be used
                rate, nsamps, score = pickle.load(f)
                
                # parse song if length is above sample_length
                n = score.shape[0]
                if n >= sample_length:
                    # Remove first melodic voice from score and reshape
                    melody = score[:,0].reshape(n)
                    
                    # Remove first m segments of score that are exactly the
                    # length we need (sample_length)
                    m = n // sample_length
                    for i in range(m):
                        # append segment to dataset
                        dataset.append(melody[sample_length*i:sample_length*(i+1)])
        
        # Vectorize the dataset
        val_dataset = []
        for data in dataset:
            new_data = np.zeros(data.shape,np.int32)
            for i in range(data.shape[0]):
                new_data[i] = labels2int_map[data[i]]
            new_data = new_data.reshape(measures, measure_len)
            val_dataset.append(new_data)
    
        # Convert list to single numpy array
        val_dataset = np.array(val_dataset)
            
        
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
    
    