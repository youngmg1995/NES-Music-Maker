# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 22:08:58 2020

@author: Mitchell

nesm_loader.py
~~~~~~~~~~~~~~
This file contains various functions for parsing and loading in the data
necessary for training and validating our model.
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
    dataset : list of numpy arrays
        Training data for fitting our model. Contains seprsco data for 4,502
        NES music tracks. Pre-parsed and vectorized to integer representation.
    labels2int_maps : list of dictionaries
        Dictionaries used to map each feature of our training data to integers.
    int2labels_maps : list of numpy arrays
        Inverse mappings of labels2int_maps. Used to map our training data back
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
    dataset : list of numpy arrays
        Training data for fitting our model. Contains seprsco data for 4,502
        NES music tracks. Pre-parsed and vectorized to integer representation.
    labels2int_maps : list of dictionaries
        Dictionaries used to map each feature of our training data to integers.
    int2labels_maps : list of numpy arrays
        Inverse mappings of labels2int_maps. Used to map our training data back
        from integers to original format.

    '''
    ### load all the files
    song_files = glob.glob(training_foldername+'*')
    
    # Shuffle our song files so similar songs aren't all in a row
    random.shuffle(song_files)

    # grabbing shape from first file to we can create start/end note
    with open(song_files[0], 'rb') as f:
        rate, nsamps, seprsco = pickle.load(f)

    # datapoint used to indicate start/end of each track
    start_end = np.zeros((1,seprsco.shape[1]+1),np.uint8)
    
    # Initiate start of our dataset
    dataset = [start_end[:,i] for i in range(start_end.shape[1])]
    
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
            rate, nsamps, seprsco = pickle.load(f)
            
            # Add new feature to each note in track
            new_feature = np.ones((seprsco.shape[0],1),np.uint8)
            seprsco = np.concatenate((seprsco,new_feature),1)
            
            # Add start/end note to end of track and reformat
            seprsco = np.concatenate((seprsco, start_end))
            seprsco = [seprsco[:,i] for i in range(seprsco.shape[1])]
            
            # Append track to dataset
            dataset = [np.concatenate((dataset[i],seprsco[i])) 
                       for i in range(len(dataset))]
            
        # Print update everytime 10% of files loaded
        counter += 1 
        if counter / interval_length >= interval:
            print('Files Loaded: {}%'.format(interval*10))
            interval += 1
            
    # Loading Complete and return datasets
    print('Loading Training Files Complete')
    
    # Vectorize the dataset
    dataset , labels2int_maps , int2labels_maps = vectorizer(dataset)
    
    # Save transformed dataset and mappings if filename given
    if save_filename:
        print('Saving Transformed Training Data and Mappings to Filename "{}"'.format(save_filename))
        data = {"dataset": [d.tolist() for d in dataset],
                "unique_values": [m.tolist() for m in int2labels_maps]
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
    dtypes : list of dtypes, optional
        List of dtypes used to specify the dtype of each feature in the
        original training dataset. These may be necessary to specify because
        certain dtypes are not the implicit dtypes loaded from our json
        save_file. The default is None.

    Returns
    -------
    dataset : list of numpy arrays
        Training data for fitting our model. Contains seprsco data for 4,502
        NES music tracks. Pre-parsed and vectorized to integer representation.
    labels2int_maps : list of dictionaries
        Dictionaries used to map each feature of our training data to integers.
    int2labels_maps : list of numpy arrays
        Inverse mappings of labels2int_maps. Used to map our training data back
        from integers to original format.

    '''
    print('Reloading Pre-Transformed Training Data from Filename "{}"'.format(save_filename))
    
    # Load in data with json
    with open(save_filename, 'r') as f:
        data = json.load(f)
        
    # Build default dtypes array if not prescribed
    # For this project, our 8-bit song data needs to all be np.uint8
    if dtypes == None:
        dtypes = []
        for i in range(len(data["dataset"])):
            dtypes.append(np.uint8)
    
    # Build datasets and mappings
    dataset = [np.array(data["dataset"][i])
               for i in range(len(data["dataset"]))]
    
    labels2int_maps = [{u:i for i, u in enumerate(np.array(data["unique_values"][i],dtypes[i]))}
                       for i in range(len(data["unique_values"]))]
    
    int2labels_maps = [np.array(data["unique_values"][i], dtypes[i])
                  for i in range(len(data["unique_values"]))]
    # Print that we are done
    print('Loading Training Data Complete')
    
    return dataset , labels2int_maps , int2labels_maps


def vectorizer(dataset):
    '''
    Functon that transforms our dataset from arbitrary labels to integers. In
    doing so it indentifies the unique lables/values for each feature in the
    dataset. These unique values are used to create the mappings fom lables
    to integers as well as the inverse mappings.

    Parameters
    ----------
    dataset : list of numpy arrays
        Input dataset is assumed to be a list with each value in the list
        containing the data for a single input feature for our dataset. The
        feature datasets are assumed to be numpy arrays with shape (N,). This
        is format our loaded training seprsco data in the load_training
        function.

    Returns : 
    -------
    vectorized_dataset : list of numpy arrays
        Transformed dataset where each feature datapoint is now represented
        using an integer. Has same shape/structure as originial input dataset.
    
    labels2int_dicts : list of dictionaries
        List containing the transformation dictionaries used to vectorize
        each feature in the input dataset. Must apply dictionaries in loop(s)
        over dataset to transform.
        
    int2labels_dicts : list of numpy arrays
        List containing numpy arrays that act as dictionaries for transforming
        our dataset back from to its original label format. In this format the
        entire integer feature array can be transformed back to its orginal
        form by passing the whole array (no need to loop over data).
    '''
    # Initialize lists for each output
    vectorized_dataset , labels2int_maps , int2labels_maps = [] , [] , []
    
    # Create dictionaries by looping over features and grabbing unique labels
    for data in dataset:
        # Find unique labels
        unique_labels = sorted(set(data))
        # Create transformation mappings
        labels_map = {u:i for i, u in enumerate(unique_labels)}
        int_map = np.array(unique_labels)
        # Append transformation mappings to corresponding lists
        labels2int_maps.append(labels_map)
        int2labels_maps.append(int_map)
    
    # Parameters used during loop
    counter = 0
    N = len(dataset)
    
    # Loop over data and vectorize to ints
    print('Vectorizing Training Data')
    for i in range(len(dataset)):
        data = dataset[i].copy()
        new_data = np.zeros(data.shape,np.int32)
        mapping = labels2int_maps[i]
        for j in range(data.shape[0]):
            new_data[j] = mapping[data[j]]
        
        # Append to output dataset
        vectorized_dataset.append(new_data)
        
        # Print progress
        counter += 1
        print('Data Vectorized: {}%'.format(int(counter/N*100)))
    
    print('Vectorizing Complete')
    
    return vectorized_dataset , labels2int_maps , int2labels_maps


def load_validation(validation_folder, labels2int_maps, save_filename=None):
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
        val_dataset = [np.array(data["val_dataset"][i])
                   for i in range(len(data["val_dataset"]))]
        
    else:
        print('Loading Validation Data from Directory: "{}"'.format(validation_folder))
        
        ### load all the files
        song_files = glob.glob(validation_folder+'*')
        
        # Shuffle our song files so similar songs aren't all in a row
        random.shuffle(song_files)
    
        # grabbing shape from first file to we can create start/end note
        with open(song_files[0], 'rb') as f:
            rate, nsamps, seprsco = pickle.load(f)
    
        # datapoint used to indicate start/end of each track
        start_end = np.zeros((1,seprsco.shape[1]+1),np.uint8)
        
        # Initiate start of our dataset
        dataset = [start_end[:,i] for i in range(start_end.shape[1])]
        
        # Initiate loop to load in data, combining all tracks into one long track
        for file in song_files:
            # Open file
            with open(file, 'rb') as f:
                # Load in data from file. Only seprsco data will be used
                rate, nsamps, seprsco = pickle.load(f)
                
                # Add new feature to each note in track
                new_feature = np.ones((seprsco.shape[0],1),np.uint8)
                seprsco = np.concatenate((seprsco,new_feature),1)
                
                # Add start/end note to end of track and reformat
                seprsco = np.concatenate((seprsco, start_end))
                seprsco = [seprsco[:,i] for i in range(seprsco.shape[1])]
                
                # Append track to dataset
                dataset = [np.concatenate((dataset[i],seprsco[i]))
                               for i in range(len(dataset))]
        
        # Vectorize the dataset
        val_dataset = []
        for i in range(len(dataset)):
            data = dataset[i].copy()
            new_data = np.zeros(data.shape,np.int32)
            mapping = labels2int_maps[i]
            for j in range(data.shape[0]):
                new_data[j] = mapping[data[j]]
            
            # Append to output dataset
            val_dataset.append(new_data)
        
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
    

def load_track(song_filename, labels2int_maps):
    '''
    Function used for loading a single seprsco file. Similar to
    load_validation function, but only loads a single file and has no option
    to save the parsed data.

    Parameters
    ----------
    song_filename : string
        Filename specifying location of seprsco file to load and parse.
    labels2int_maps : list of dictionaries
        List containing the transformation mappings used to vectorize the
        training data. These are also used to vectorize the track.

    Returns
    -------
    track_data : list of numpy arrays
        Parsed data for the given track that is ready to feed into our model.

    '''
    # Load in data from file. Only seprsco data will be used
    with open(song_filename, 'rb') as f:
        rate, nsamps, seprsco = pickle.load(f)
    
    # Add new feature to each note in track
    new_feature = np.ones((seprsco.shape[0],1),np.uint8)
    seprsco = np.concatenate((seprsco,new_feature),1)
    
    # Reshape
    seprsco = [seprsco[:,i] for i in range(seprsco.shape[1])]
    
    # Vectorize the data
    track_data = []
    for i in range(len(seprsco)):
        data = seprsco[i].copy()
        new_data = np.zeros(data.shape,np.int32)
        mapping = labels2int_maps[i]
        for j in range(data.shape[0]):
            new_data[j] = mapping[data[j]]
        
        # Append to output dataset
        track_data.append(new_data)
            
    return track_data
    
    