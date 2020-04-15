# NES-Music-Maker
### Using Neural Networks to Generate 8-Bit Music for the Nintendo Entertainment System

<img src="/xstatic/explosion8bit4-8.2.2017.jpg" width=800>

## Overview
In this project we attempt to create our own original NES soundtracks with the help of neural networks. The music we generate will both mimic the NES soundtracks we train against, and will be formatted properly for conversion and play on a standard Nintendo Entertainment System audio synthesizer. This means our music will not only sound like 8-bit video game music, but will actually be 8-bit video game music!!! 
In order to accomplish this task we did the following:
1) Loaded and parsed a dataset of 5,000+ NESM soundtrack files
2) Designed various LSTM and VAE models that could learn from and reproduce these soundtracks
3) Trained each model against the soundtracks in our training dataset
4) Generated new NESM soundtracks using our trained models and an initialization seed

## NESM Dataset
To train and validate all our models, we will be using a dataset of 5,278 songs from 397 NES games. This dataset comes from [The NES Music Database](https://github.com/chrisdonahue/nesmdb) which has done a majority of the work compiling and formatting the NESM soundtracks in a manner that is easy to analyse and convert. In addition to the music files, this database also offers a python library filled with functions for converting the soundtracks to other formats useful for listening to the music and playing it on the NES audio synth. Since this library is written in Python 2.X and we use Python 3.X, included in our repository is a modified subset of Chris's nesmdb package that we use for converting to the formats we need.
The NESMDB offers numerous formatts for the NES music, each with their own strangths and weaknesses for modeling against. For our purposes, we chose to use the Separated Score (seprsco) format. This format is lossy and does not include all the performance features of the NES music, but it does include all the note information for the music and is easier to parse given its simple array format.

<img src="/xstatic/score_separated.png" width=600>

Excluding the playback channel, the NES audio synth has 4 instrument voices: 2 pulse-wave generators (P1, P2), a triangle-wave generator (TR), and a percussive noise generator (NO). The Seperated Score format contains a piano roll representation of each of these intrument voices created by sampling the music at a fixed rate of 24Hz. As mentioned before, this makes our music tracks lossy, but gives them a very standardized structure that is easy to manipulate. 

<img src="/xstatic/seprsco.png" width=400>

While this dataset is already very structured, we further parse and format our dataset using functions in the included 'dataset_utils.py' files. These functions load in the seprsco formatted files, vectorize them so all notes are integer representations from a connected range, [0, 1, ..., N], and reshape our dataset to fit the input specifications of each model.

## Model Designs


## Training


## Soundtrack Generation


## Results


## Acknowledgements 
