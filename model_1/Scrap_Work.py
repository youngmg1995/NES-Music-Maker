# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 19:08:25 2020

@author: Mitchell
"""

import glob, pickle
import numpy as np
import matplotlib.pyplot as plt


training_folder = '../nesmdb24_seprsco/train/*'
song_files = glob.glob(training_folder)

# pick one at random file
song_file_0 = np.random.choice(song_files)
print(song_file_0)

with open(song_file_0, 'rb') as f:
  rate, nsamps, seprsco = pickle.load(f)

print('Temporal discretization rate: {}'.format(rate)) # Will be 24.0
print('Length of original VGM: {}'.format(nsamps / 44100.))
print('Piano roll shape: {}'.format(seprsco.shape))
print('What I Need')
print(nsamps / seprsco.shape[0])

# Finding average nsamps / seprsco.shape[0]
training_folder = '../nesmdb24_seprsco/train/*'
song_files = glob.glob(training_folder)

# for storing values
samples = np.zeros(len(song_files))

# loop over all files
for i in range(len(song_files)):
    with open(song_files[i], 'rb') as f:
      rate, nsamps, seprsco = pickle.load(f)
      samples[i] = nsamps / seprsco.shape[0]

print('Median: {}'.format(np.median(samples)))
print('Mean: {}'.format(np.mean(samples)))
print('STD: {}'.format(np.std(samples)))
plt.hist(samples)

# Sample of converting seprsco to waveform
import pickle
from nesmdb.convert import seprsco_to_wav
filename = "C:/Users/Mitchell/Documents/Studies/AI/AI-8-Bit-Nintendo-Music/nesmdb24_seprsco/train/076_DonkeyKongCountry4_00_01Theme.seprsco.pkl"
seprsco = None
with open(filename, 'rb') as f:
  seprsco = pickle.load(f)
wav = seprsco_to_wav(seprsco)


vgm_check_2 = "C:/Users/Mitchell/Documents/Studies/AI/AI-8-Bit-Nintendo-Music/vgm_check_2"


