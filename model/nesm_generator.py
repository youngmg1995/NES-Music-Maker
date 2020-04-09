# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 17:14:19 2020

@author: Mitchell
"""

# Imports
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# NOTE - nesmdb folder manually added to environment libraries 
from model_builder import model_builder
from generation_utils import generate_track, generate_seprsco
from nesm_loader import load_training, load_track
import nesmdb
from nesmdb.vgm.vgm_to_wav import save_vgmwav
import tensorflow as tf
import numpy as np
import os


### Load Mappings
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
data_foldername = '../nesmdb24_seprsco/train/'
save_filename = 'transformed_dataset.json'
dataset , labels2int_maps , int2labels_maps = load_training(data_foldername, save_filename)
# Delete dataset to free up memory
del dataset


### Reinitiate Model
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Define Model Structure
model = model_builder(rnn_units = 1024,
                        input_dims = [len(v) for v in int2labels_maps], 
                        emb_output_dims = [128, 128, 32, 16],
                        batch_size = 1
                        )

# Reload Saved Weights
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "model_ckpt")
model.load_weights(checkpoint_prefix)
model.build(tf.TensorShape([1, None, 5]))

# Print Summary of Model
model.summary()


### Create Training Seeds
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Create two types of seeds for generating NES music
#   1) Using indicater for signifying start/end of each track
#   2) Using random segment from chosen track

# Build Seed 1
start_seed = [np.zeros((1,1),int) for i in range(5)]

# Build Seed 2
seed_length = 50
song_filename = "../nesmdb24_seprsco/train/322_SuperMarioBros__00_01RunningAbout.seprsco.pkl"
song_data = load_track(song_filename, labels2int_maps)
random_idx = np.random.choice(song_data[0].shape[0]-seed_length)
my_seed = [song_data[i][random_idx:random_idx+seed_length].reshape((1,seed_length)) 
           for i in range(5)]


### Generate NES Tracks
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Generation Parameters
generation_length = 2000
num_songs = 3

# Generating Tracks Using Seed 1 and Method 1 (set # of tracks generated)
print('Generating track with {} songs using start seed'.format(num_songs))
random_track = generate_track(model, start_seed, num_songs = num_songs,
                              gen_method = 1)

# Generating Tracks Using Seed 2 and Method 0 (set length of generated track)
print('Generating track of length {} using random song seed'.\
      format(generation_length))
my_track = generate_track(model, my_seed,
                          generation_length = generation_length,
                          gen_method = 0)


### Parse Valid Tracks
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set minimum valid track length
min_length = 100

# Parse tracks into seprsco data
print('Converting generated track to valid seprsco tracks with length > {}.'\
      .format(min_length))
random_tracks = generate_seprsco(random_track, int2labels_maps, min_length)
my_tracks = generate_seprsco(my_track, int2labels_maps, min_length)


### Convert to WAV Files
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Transforming generated songs into waveforms so we can listen!!!
print('Converting spersco tracks to WAV audio tracks.')
random_wavs , my_wavs = [] , []
for track in random_tracks:
    wav = nesmdb.convert.seprsco_to_wav(track)
    random_wavs.append(wav)
for track in my_tracks:
    wav = nesmdb.convert.seprsco_to_wav(track)
    my_wavs.append(wav)


### Save WAV Files
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Save our wav tracks to appropriate files (be sure not to overwrite existing)
print('Saving generated WAV audio tracks.')
wav_folder = 'model_gen_files/wav/'
for i in range(len(random_wavs)):
    random_file = wav_folder+'my_wav_{}.wav'.format(i)
    save_vgmwav(random_file, random_wavs[i])
for i in range(len(my_wavs)):
    my_file = wav_folder+'random_wav_{}.wav'.format(i)
    save_vgmwav(my_file, my_wavs[i])


