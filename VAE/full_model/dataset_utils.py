# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 22:08:58 2020

@author: Mitchell

dataset_utils.py
~~~~~~~~~~~~~~~~
This file contains various functions for parsing and loading in our datasets.
These datasets will be used for training our VAE model, validatig the model,
and generating new NES soundtracks using our model.

Functions contained include the following:
    
    1) load_training
    - Wrapper function called for loading in training dataset. Calls
    load_seprsco or reload_data to do so.
    
    2) load_seprsco
    - Main function used to load in training data from folder of seprsco
    NESM files. Reads files, parses data, and saves transformed data all in
    one.
    
    3) reload_data
    - Loads training data from file location containing pre-parsed data
    from previous training session. (No need to redo work)
    
    4) vectorizer
    - Transforms training data to integer representations that are simpler to
    train. Returns trasformed data as well as mappings for transformation and
    inverse mappings.
    
    5) load_validation
    - Similar to load_seprsco and reload data, but loads validation dataset
    instead of training dataset. To do so, requires vectorization mappings
    obtained from loading the training data.
    
"""

import glob, pickle, json
import numpy as np
import os, random


def load_training(training_foldername, save_filename=None,
                  measures = 8, measure_len = 96):
    '''
    Function that loads training data from prescribed folder or from given 
    savefile location if it already exists. Dataset returned is in the specific
    format required for training the VAE class models defined in this folder.

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
    dataset : list of numpy arrays
        Training data for fitting our VAE model. The dataset contains 4 numpy
        arrays, where each array represents a different voice from the NES
        music. They are seperated since our model treats each as a separate
        input feature. Shape of each is [# tracks, measures, measure_length].
    labels2int_maps : list of dictionaries
        Dictionaries used to map original training data to integers. There is a
        different mapping for each voice/feature in our dataset.
    int2labels_maps : list of numpy arrays
        Inverse mappings of labels2int_maps. Used to map our training data back
        from integers to original format. One mapping for each voice/feature in
        our dataset.

    '''
    # Check if save_filename exists. 
    # If so load pre-transformed data from there using 'reload_data'.
    # If not, load orginal seprsco files using 'load_seprsco'.
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
    dataset : list of numpy arrays
        Training data for fitting our VAE model. The dataset contains 4 numpy
        arrays, where each array represents a different voice from the NES
        music. They are seperated since our model treats each as a separate
        input feature. Shape of each is [# tracks, measures, measure_length].
    labels2int_maps : list of dictionaries
        Dictionaries used to map original training data to integers. There is a
        different mapping for each voice/feature in our dataset.
    int2labels_maps : list of numpy arrays
        Inverse mappings of labels2int_maps. Used to map our training data back
        from integers to original format. One mapping for each voice/feature in
        our dataset.

    '''
    # Set sample_length of samples based on measures and measure_len
    sample_length = measures * measure_len
    
    # load all the filenames
    song_files = glob.glob(training_foldername+'*')
    
    # Shuffle our song filenames so similar songs aren't all in a row
    random.shuffle(song_files)
    
    # Initiate start of our dataset as list of 4 lists (1 for each voice in track)
    dataset = [[] for i in range(4)]
    
    # Parameters printing progress of loading in data
    counter = 0
    interval = 1
    interval_length = len(song_files) / 10
    
    # Initiate loop to load in data
    print('Loading Training Data from Directory: "{}"'.format(training_foldername))
    for file in song_files:
        # Open file
        with open(file, 'rb') as f:
            # Load in data from file. Only score data will be used
            rate, nsamps, score = pickle.load(f)
            
            # Only use soundtracks longer than our sample_length
            m , n = score.shape
            if m >= sample_length:
                # Remove first m segments of score that are exactly the
                # length we need (sample_length)
                M = m // sample_length
                for i in range(M):
                    # iterate over each voice of the score
                    for j in range(n):
                        # append segment to dataset
                        dataset[j].append(score[sample_length*i:sample_length*(i+1), j])
            
        # Print update everytime 10% of files loaded
        counter += 1 
        if counter / interval_length >= interval:
            print('Files Loaded: {}%'.format(interval*10))
            interval += 1
            
    # Loading Complete and return datasets
    print('Loading Training Files Complete')
    
    # Vectorize the dataset
    dataset , labels2int_maps , int2labels_maps = vectorizer(dataset, measures, measure_len)
    
    # Save transformed dataset and mappings if filename given
    if save_filename:
        print('Saving Transformed Training Data and Mappings to Filename "{}"'.format(save_filename))
        data = {"dataset": [d.tolist() for d in dataset],
                "unique_values": [mapping.tolist() for mapping in int2labels_maps]
                }
        with open(save_filename, 'w') as f:
            json.dump(data, f)
        print('Saving Data Complete')

    # Print that we are done
    print('Loading Training Data Complete')
    
    return dataset , labels2int_maps , int2labels_maps


def reload_data(save_filename, dtypes = None):
    '''
    Function that loads pre-parsed training data from specified save_filename.

    Parameters
    ----------
    save_filename : string
        Filename specifying location of pre-parsed training data.
    dtypes : dtypes, optional
        Used to specify the dtypes of each feature in our original training
        dataset. This may be necessary to specify because certain dtypes are
        not the implicit dtypes loaded from our json save_file. The default is
        set to np.uint8 which is the dtype for all data in our seprsco scores.

    Returns
    -------
    dataset : list of numpy arrays
        Training data for fitting our VAE model. The dataset contains 4 numpy
        arrays, where each array represents a different voice from the NES
        music. They are seperated since our model treats each as a separate
        input feature. Shape of each is [# tracks, measures, measure_length].
    labels2int_maps : list of dictionaries
        Dictionaries used to map original training data to integers. There is a
        different mapping for each voice/feature in our dataset.
    int2labels_maps : list of numpy arrays
        Inverse mappings of labels2int_maps. Used to map our training data back
        from integers to original format. One mapping for each voice/feature in
        our dataset.

    '''
    print('Reloading Pre-Transformed Training Data from Filename "{}"'.format(save_filename))
    
    # Load in data with json
    with open(save_filename, 'r') as f:
        data = json.load(f)
        
    # Build default dtypes array if not prescribed
    # For this project, our 8-bit song data needs to all be np.uint8
    if dtypes == None:
        dtypes = []
        for i in range(len(data["unique_values"])):
            dtypes.append(np.uint8)
    
    # Build datasets and mappings
    dataset = [np.array(d) for d in data["dataset"]]
    
    labels2int_maps = [{u:i for i, u in enumerate(np.array(mapping, dtype))}
                      for dtype, mapping in zip(dtypes, data["unique_values"])]
    
    int2labels_maps = [np.array(mapping, dtype)
                      for dtype, mapping in zip(dtypes, data["unique_values"])]
    
    # Print that we are done
    print('Loading Training Data Complete')
    
    return dataset , labels2int_maps , int2labels_maps


def vectorizer(dataset, measures = 8, measure_len = 96):
    '''
    Functon that transforms our dataset from arbitrary labels to integers. In
    doing so it indentifies the unique lables/values in our dataset. These
    unique values are used to create the mappings fom lables to integers as
    well as the inverse mappings.

    Parameters
    ----------
    dataset : list of lists of numpy arrays
        Dataset we are vectorizing. Assumed to be list of lists of numpy
        arrays. The inner lists are different features of our dataset, and each
        numpy array within these lists represent a training sample.
    measures : int
        Outermost dimension used for reshaping our vectorized dataset samples.
        The default is 8.
    measure_len : int
        Innermost dimension used for reshaping our vectorized dataset samples.
        The default is 96.

    Returns : 
    -------
    vectorized_dataset : list of numpy arrays
        Transformed dataset where each datapoint is now represented using an
        integer. Has same number of features as input dataset, but now
        each feature is combined to single array with shape
        [# tracks, measures, measure_length].
    labels2int_maps : list of dictionaries
        Dictionaries used to map original training data to integers. There is a
        different mapping for each voice/feature in our dataset.
    int2labels_maps : list of numpy arrays
        Inverse mappings of labels2int_maps. Used to map our training data back
        from integers to original format. One mapping for each voice/feature in
        our dataset.
        
    '''
    print('Vectorizing Training Data')
    
    # Initialize lists for storing mappings
    labels2int_maps , int2labels_maps = [] , []
    
    # Loop over data to find unique labels and create mappings
    for i in range(len(dataset)):
        # Find unique labels
        unique_labels = sorted(set(np.concatenate(dataset[i])))
        # Create mappings
        labels2int_maps.append({u:i for i, u in enumerate(unique_labels)})
        int2labels_maps.append(np.array(unique_labels))
    
    # Loop over data and vectorize to ints using mapping and reshape
    vectorized_dataset = [[] for i in range(len(int2labels_maps))]
    for i in range(len(int2labels_maps)):
        mapping = labels2int_maps[i]
        for data in dataset[i]:
            new_data = np.zeros(data.shape, np.int32)
            for j in range(data.shape[0]):
                # Vectorize
                new_data[j] = mapping[data[j]]
            # Reshape
            new_data = new_data.reshape(measures, measure_len)
            # Append to vectorized_dataset
            vectorized_dataset[i].append(new_data)
        
        # Convert each feature to single numpy array
        vectorized_dataset[i] = np.array(vectorized_dataset[i])
    
    print('Vectorizing Complete')
    
    return vectorized_dataset , labels2int_maps , int2labels_maps


def load_validation(validation_folder, labels2int_maps, save_filename=None,
                    measures = 8, measure_len = 96):
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
    val_dataset : list of numpy arrays
        Validation data for testing our VAE model. The dataset contains 4 numpy
        arrays, where each array represents a different voice from the NES
        music. They are seperated since our model treats each as a separate
        input feature. Shape of each is [# tracks, measures, measure_length].

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
        val_dataset = [np.array(d) for d in data["val_dataset"]]
        
    else:
        print('Loading Validation Data from Directory: "{}"'.format(validation_folder))
        
        # Set sample_length of samples based of measures and measure_len
        sample_length = measures * measure_len
        
        # load all the filenames
        song_files = glob.glob(validation_folder+'*')
        
        # Shuffle our song files so similar songs aren't all in a row
        random.shuffle(song_files)
        
        # Initiate list for dataset
        dataset = [[] for i in range(4)]
        
        # Initiate loop to load in data for each file
        for file in song_files:
            # Open file
            with open(file, 'rb') as f:
                # Load in data from file. Only score data will be used
                rate, nsamps, score = pickle.load(f)
                
                # Only parse songs above sample_length
                m , n = score.shape
                if m >= sample_length:
                    # Remove first m segments of score that are exactly the
                    # length we need (sample_length)
                    M = m // sample_length
                    for i in range(M):
                        # iterate over each voice of the score
                        for j in range(n):
                            # append segment to dataset
                            dataset[j].append(score[sample_length*i:sample_length*(i+1), j])
        
        # Vectorize the dataset
        val_dataset = [[] for i in range(len(labels2int_maps))]
        for i in range(len(labels2int_maps)):
            mapping = labels2int_maps[i]
            for data in dataset[i]:
                new_data = np.zeros(data.shape, np.int32)
                for j in range(data.shape[0]):
                    # Vectorize
                    new_data[j] = mapping[data[j]]
                # Reshape
                new_data = new_data.reshape(measures, measure_len)
                # Append to vectorized_dataset
                val_dataset[i].append(new_data)
            
            # Convert each feature to single numpy array
            val_dataset[i] = np.array(val_dataset[i])
        
        
        # Save transformed dataset if filename given
        if save_filename:
            print('Saving Transformed Validation Data to Filename "{}"'.format(save_filename))
            data = {"val_dataset": [d.tolist() for d in val_dataset]}
            with open(save_filename, 'w') as f:
                json.dump(data, f)
            print('Saving Data Complete')
    
    # Print that we are done
    print('Loading Validation Data Complete')
    
    return val_dataset
    
    