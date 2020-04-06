# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 22:08:58 2020

@author: Mitchell

preprocessor.p
~~~~~~~~~~~~~~

"""
import glob, pickle, json
import numpy as np
import os, random

# Function used to load data for our model
def load_data(data_foldername, save_filename=None):
    '''
    '''
    # Check if savefile exists. 
    # If so load pre-transormed data from there using 'reload_data'.
    # If not, load orginal seprsco files using 'initial_load_data'.
    if save_filename and os.path.exists(save_filename):
        dataset , labels2int_maps , int2labels_maps = reload_data(save_filename)
    else:
        dataset , labels2int_maps , int2labels_maps = load_seprsco(data_foldername, save_filename)
    
    return dataset , labels2int_maps , int2labels_maps
        
     

# Function that loads and transforms seprsco files if a saved version
# doesn't already exist
def load_seprsco(data_foldername, save_filename=None):
    '''
    '''
    ### load all the files
    song_files = glob.glob(data_foldername+'*')
    
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
    print('Loading Seprsco Files from Directory: "{}"'.format(data_foldername))
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
    print('Loading Seprsco Files Complete')
    
    # Vectorize the dataset
    dataset , labels2int_maps , int2labels_maps = vectorizer(dataset)
    
    # Save transformed dataset and mappings if filename given
    if save_filename:
        print('Saving Transformed Dataset and Mappings to Filename "{}"'.format(save_filename))
        data = {"dataset": [d.tolist() for d in dataset],
                "unique_values": [m.tolist() for m in int2labels_maps]
                }
        with open(save_filename, 'w') as f:
            json.dump(data, f)
        print('Saving Dataset Complete')

    # Print that we are done
    print('Loading Dataset Complete')
    
    return dataset , labels2int_maps , int2labels_maps

def reload_data(save_filename, dtypes = None):
    '''
    '''
    print('Reloading Pre-Transformed Dataset from Filename "{}"'.format(save_filename))
    
    
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
    print('Loading Dataset Complete')
    
    return dataset , labels2int_maps , int2labels_maps

# Function for vectorizing our dataset from labels to integers
def vectorizer(dataset):
    '''
    Functon that finds set of labels for each feature. Then transforms the
    dataset from these labels to integer represntations. Returns the
    transformed data as well as dictionaries that can be used to vectorize
    and de-vectorize the data in the future.

    Parameters
    ----------
    dataset : list of numpy arrays
        Input dataset is assumed to be a list with each value in the list
        containing the data for a single input feature for our dataset. The
        feature datasets are assumed to be numpy arrays with shape (N,)

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
    print('Vectorizing Seprsco Dataset')
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

def load_song(song_filename):
    '''
    '''
    # Load in data from file. Only seprsco data will be used
    with open(song_filename, 'rb') as f:
        rate, nsamps, seprsco = pickle.load(f)
    
    # Add new feature to each note in track
    new_feature = np.ones((seprsco.shape[0],1),np.uint8)
    seprsco = np.concatenate((seprsco,new_feature),1)
    
    # Reshape
    seprsco = [seprsco[:,i] for i in range(seprsco.shape[1])]
    
    return seprsco
    
    