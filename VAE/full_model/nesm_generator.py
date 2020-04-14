# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 17:14:19 2020

@author: Mitchell

nesm_generator.py
~~~~~~~~~~~~~~~~~
This file serves as a script for using our pre-trained VAE model to generate
brand new NES music soundtracks. To do so we first reconstruct our model using
the file VAE class defined in `VAE.py` and the same parameters used in 
`model_training`. Then we use functions from the file `generation_utils` to 
have our trained model create entirely new and original NES music.
"""

# Imports
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# NOTE - nesmdb folder manually added to environment libraries 
from dataset_utils import load_training
from VAE import VAE
from generation_utils import generate_seprsco, latent_SVD
import nesmdb
from nesmdb.vgm.vgm_to_wav import save_vgmwav
import tensorflow as tf
import numpy as np
import os, json


### Load Mappings
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Parameters for shape of dataset (note these are also used for model def.)
measures = 8
measure_len = 96

# load data
training_foldername = '../../nesmdb24_seprsco/train/'
train_save_filename = 'transformed_dataset.json'
dataset , labels2int_maps , int2labels_maps = \
    load_training(training_foldername, train_save_filename,
                  measures = measures, measure_len = measure_len)


### Reinitiate Model
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
### Model Parameters
latent_dim = 124
input_dims = [mapping.shape[0] for mapping in int2labels_maps]
embed_dims = [32, 32, 32, 8]
dropout = .1
maxnorm = None
vae_b1 , vae_b2 = .02 , .1

print('Reinitiating VAE Model')

# Build Model
model = VAE(latent_dim, input_dims, embed_dims, measures, measure_len, dropout, 
            maxnorm, vae_b1 , vae_b2)

# Reload Saved Weights
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "model_ckpt")
model.load_weights(checkpoint_prefix)
model.build([tf.TensorShape([None, measures, measure_len]) for i in range(4)])

# Print Summary of Model
model.summary()


### Sample Latent Variable Space
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Here we use SVD to more effectively sample from the orthogonal components
# of the distribution over our latent space

# Parameters for sampling
num_songs = 10

print('Generating Latent Samples to Generate {} New Tracks'.format(num_songs))

# Grab distributions of dataset over latent space
latent_vecs = model.reparameterize(model.encoder(dataset))

# Sample from normal distribution
rand_vecs = np.random.normal(0.0, 1.0, (num_songs, latent_dim))

# perform SVD
plot_eigenvalues = True
sample_vecs = latent_SVD(latent_vecs, rand_vecs, plot_eigenvalues)


### Generate New Tracks
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Create new seprsco tracks using our model and the random samples
# Seprsco files can later be converted to valid NES music format

print('Generating New Tracks from Latent Samples')

# Decode samples using VAE
decoded_tracks = model.decoder(sample_vecs)

# Pull most likely note for each step in track for each feature voice
decoded_tracks = [tf.math.argmax(feature, axis=-1).numpy()
                  for feature in decoded_tracks]

print('Converting Model Output to Seprsco')

# Convert tracks to seprsco format
seprsco_tracks = generate_seprsco(decoded_tracks, int2labels_maps)


### Convert to WAV
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Convert seprsco tracks to WAV files so we can listen!!!

print('Converting Seprsco WAV Audio')
wav_tracks = []
for track in seprsco_tracks:
    wav = nesmdb.convert.seprsco_to_wav(track)
    wav_tracks.append(wav)


### Save WAV Files
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Save our wav tracks to appropriate files (be sure not to overwrite existing)
# Also save latent variables so we can reproduce songs we like

# Save WAV tracks
save_wav = True
if save_wav:
    print('Saving Generated WAV Audio Tracks')
    wav_folder = 'model_gen_files/'
    for i in range(len(wav_tracks)):
        wav_file = wav_folder+'VAE_NESM_{}.wav'.format(i)
        save_vgmwav(wav_file, wav_tracks[i])

# Save Latent Variables
save_latent_var = True
if save_latent_var:
    print('Saving Latent Variables for Generated Tracks')
    latent_filename = os.path.join(wav_folder, "latent_variables.json")
    with open(latent_filename, 'w') as f:
        json.dump({
            'VAE_NESM_{}.wav'.format(i): sample_vecs[i].tolist()
            for i in range(sample_vecs.shape[0])
            }, f)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#----------------------------------END FILE------------------------------------
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~