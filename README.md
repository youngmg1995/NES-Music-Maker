# NES-Music-Maker

### Using Neural Networks to Generate 8-Bit Music for the Nintendo Entertainment System

<p align="center"><img src="/xstatic/explosion8bit4-8.2.2017.jpg" width=800></p>

## Overview

In this project we attempt to create our own original NES soundtracks with the help of neural networks. The music we generate will both mimic the NES soundtracks we train against, and will be formatted properly for conversion and play on a standard Nintendo Entertainment System audio synthesizer. This means our music will not only sound like 8-bit video game music, but will actually be 8-bit video game music!!!

In order to accomplish this task we did the following:

1) Loaded and parsed a dataset of 5,000+ NESM soundtrack files
2) Designed various LSTM and VAE models that could learn from and reproduce these soundtracks
3) Trained each model against the soundtracks in our training dataset
4) Generated new NESM soundtracks using our trained models and an initialization seed

## NESM Dataset

To train and validate all our models, we used a dataset of 5,278 songs from 397 NES games. This dataset comes from [The NES Music Database](https://github.com/chrisdonahue/nesmdb) which has done a majority of the work compiling and formatting the NESM soundtracks in a manner that is easy to analyse and convert. In addition to the music files, this database also offers a python library filled with functions for converting the soundtracks to other formats useful for listening to the music and playing it on the NES audio synth. Since this library is written in Python 2.X and we use Python 3.X, included in our repository is a modified subset of Chris's nesmdb package that we use for converting to the formats we need.

The NESMDB offers numerous formatts for the NES music, each with their own strangths and weaknesses for modeling against. For our purposes, we chose to use the Separated Score (seprsco) format. This format is lossy and does not include all the performance features of the NES music, but it does include all the note information for the music and is easier to parse given its simple array format.

<p align="center"><img src="/xstatic/score_separated.png" width=600></p>

Excluding the playback channel, the NES audio synth has 4 instrument voices: 2 pulse-wave generators (P1, P2), a triangle-wave generator (TR), and a percussive noise generator (NO). The Seperated Score format contains a piano roll representation of each of these intrument voices created by sampling the music at a fixed rate of 24Hz. As mentioned before, this makes our music tracks lossy, but gives them a very standardized structure that is easy to manipulate. 

<p align="center"><img src="/xstatic/seprsco.png" width=600></p>

While this dataset is already very structured, we further parse and format our dataset using functions in the included 'dataset_utils.py' files. These functions load in the seprsco formatted files, vectorize them so all notes are integer representations from a connected range, [0, 1, ..., N], and reshape our dataset to fit the input specifications of each model.

## Model Designs

To generate our NES soundtracks we designed and tested two different classes of neural network models:

1) **LSTM** - Long Short-Term Memory
2) **VAE** - Variational Autoencoder

For each model type we contructed two different model designs:

1) **Reduced Model** - Analyses only the 1st melodic voice (P1) of each NES track
2) **Full Model** - Analyses all 4 intrument voices (P1, P2, TR, NO) of each NES track

Most generative models for music out on the internet focus on generating a single melodic voice (Ex. classical piano pieces). This is a much simpler problem to model since it reduces the dimensionality of our inputs/outputs. Illustrating this point using NES music, the possible combinations of notes for just the P1 voice is only 77, whereas the possible combinations of notes for all 4 voices (P1, P2, TR, NO) combined is 77 x 77 x 89 x 17 = 8,970,577.

As such, we decided to follow suite with the rest of the community and first test each model class on a reduced model that only analyses the first melodic voice of each NES track (P1). Using the reduced model allowed us to more easily optimize the training prescription and hyperparemeters for each model class. Once comfortable with the reduced model, we then expanded our model classes to the much more difficult problem of analysing all 4 instrument voices and generating complete NES soundtracks.

Below we go into more detail on each model class regarding its structure and why it was chosen for this problem.

### LSTM

LSTM or Long Short-Term Memory is a type of recurrent neural network (RNN). LSTMs are the prototypical supervised network used for modeling and generating sequential data, such as found in text, music, and video; but in general, RNN's as a class are effective at modeling time-series data. What makes RNNs so well suited for modeling temporal sequences is their ability to store an internal state in memory. This internal state can then be passed as an input, along with a singular timestep from a training example, to help predict the next value in our temporal sequence. In otherwords, whereas most neural networks are strictly feedforward structures, RNNs in contrast leverage feedback connections. This allows RNNs to use information about past timesteps to make a more informed prediction on the current timestep.

What makes LSTMs in particular such an effective form of these reccurent structures is their ability to learn long-term dependencies. Standard RNN models typically don't have the ability to look back more than a few timesteps because in practice the suffer from the vanishing gradient problem. The vanishing gradient problem is an issue characterized by increasingly diminishing partial derivatives  from layer to layer in a neural network during standard backpropogation training. In essence, this makes our RNNs difficult to train because information is inefficiently passed to earlier timesteps of our model and results in smaller corrections to these parameters.

<p align="center"><img src="/xstatic/LSTM.png" width=600></p>

LSTM networks were specifically designed to overcome the vanishing gradient problem common in these models. While not completely foolproof, LSTMs tend to allow for a more unchanged flow of gradients during backpropogation. This is acheived using a combination of cells that store the LSTM states and regulator gates that control the flow of information. 

To create our LSTM models for this project we the pre-built LSTM layers in Keras and implemented them using a TensorFlow backend within a python environment. More specifically, both our reduced and full models consisted of a single LSTM layer typically with 1024 units, which was about the largest my PC would handle. Additionally, each model leveraged an emdedding layer prepending the LSTM to map our integer inputs to a dense vector representation, and a dense layer to construct the probability distribution over the possible note choices for each instrument voice. Below is the overall structure:

1) Layer 1 - Embedding: Maps our integer inputs for each instrument voice to dense vectors of chosen dimensionality
2) Layer 2 - LSTM: Bulk of our network, a recurrent neural structure with a sigmoid activation and chosen units
3) Layer 3: Dense: Typical fully connected layer with a softmax activation used to produce output distribution

In addition to the above high level structure, we also leveraged more advanced techniques sprinkled throughout our network to optimize its performance and imporve training. These include but are not limited to dropout, max-norm weight contraint, batch normalization, etc.

### VAE

VAEs or Variational Autoencoders are a specific subset of the autoencoder class of neural networks. Autoencoders are a type of unsupervised neural network that learns patterns and characteristics of a given dataset through a representation in a lower dimensional space. Typical this means autoencoders consist of two separate stages, an encoder or compressive stage and a decoder or decompressive stage. The encoder stage maps the inputs to some lower dimensional latent space while the decoder stage recontructs the initial input back from its compressed representation. These types of models are generally used for extracting un-observable characteristics in our data or for generating new data for training other models, most commonly for computer vision.

Like general autoencoders, VAEs learn a compressed representation of our input data within a latent space and reconstruct the input as closely as possible. However, what makes VAEs particularly unique is their probabilistic nature. Instead of just mapping our inputs to a vanilla vector space, VAEs map our inputs to a space of probabilistic variables. In other words, our VAE is not learning a lower dimensional representation of our inputs, instead it is learning the distribution over our latent space from which to sample our latent vectors. In practice, this is acheived by encoding our inputs to a vector of means and standard deviations that we force to follow a Gaussian distribution using a regularization term in our loss function. We then use a reparameterization to convert these parameters to a latent vector by sampling from a normal distribution, scaling the result using our standard deviations, and adding back the mean.

<p align="center"><img src="/xstatic/VAE.png" width=600></p>

At this point you might be thinking, "How is a VAE going to model music?". It is a good question to ask, since VAEs are generally used for problems related to computer vision and image processing. However, [others](https://www.youtube.com/watch?v=UWxfnNXlVy8) have succesfully used VAEs to generate music by representing the music in a format akin to an image. For instance, in the linked work by CodeParade, each song is represented as a piano roll, which when divided into measures of 96 beats each with 96 possible notes essentially becomes a visual repesentation of the song. Now instead of a temporal sequence of notes, each song is represented as a set of 16 sequential 96 x 96 images. Using my limited understanding of music theory, this representation could make sense for our NES music, since the music is very structured and repetitive where any given not depends more on the overal melody of the measures rather than the previous note or two.

Leveraging this framework, we decided to create our VAE models using the Keras library, implemented uing a Tensorflow backend in a python environment same as with the LSTM models. However, Keras does not include pre-built VAE models/layers, so we decided to construct our encoder and decoder sections using a deep network of fully connected layers. In some layers, we apply the same network recursively over each of the measures of our soundtracks to try and caprture the sequential nature of the data. Below is a more detailed list of the models layers:

1)
2)
3)

## Training


## Soundtrack Generation


## Results


## Acknowledgements 
