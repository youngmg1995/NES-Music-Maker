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
3) Layer 3 - Dense: Typical fully connected layer with a softmax activation used to produce output distribution

In addition to the above high level structure, we also leveraged more advanced techniques sprinkled throughout our network to optimize its performance and imporve training. These include but are not limited to dropout, max-norm weight contraint, batch normalization, etc.

### VAE

VAEs or Variational Autoencoders are a specific subset of the autoencoder class of neural networks. Autoencoders are a type of unsupervised neural network that learns patterns and characteristics of a given dataset through a representation in a lower dimensional space. Typical this means autoencoders consist of two separate stages, an encoder or compressive stage and a decoder or decompressive stage. The encoder stage maps the inputs to some lower dimensional latent space while the decoder stage recontructs the initial input back from its compressed representation. These types of models are generally used for extracting un-observable characteristics in our data or for generating new data for training other models, most commonly for computer vision.

Like general autoencoders, VAEs learn a compressed representation of our input data within a latent space and reconstruct the input as closely as possible. However, what makes VAEs particularly unique is their probabilistic nature. Instead of just mapping our inputs to a vanilla vector space, VAEs map our inputs to a space of probabilistic variables. In other words, our VAE is not learning a lower dimensional representation of our inputs, instead it is learning the distribution over our latent space from which to sample our latent vectors. In practice, this is acheived by encoding our inputs to a vector of means and standard deviations that we force to follow a Gaussian distribution using a regularization term in our loss function. We then use a reparameterization to convert these parameters to a latent vector by sampling from a normal distribution, scaling the result using our standard deviations, and adding back the mean.

<p align="center"><img src="/xstatic/VAE.png" width=600></p>

At this point you might be thinking, "How is a VAE going to model music?". It is a good question to ask, since VAEs are generally used for problems related to computer vision and image processing. However, [others](https://www.youtube.com/watch?v=UWxfnNXlVy8) have succesfully used VAEs to generate music by representing the music in a format akin to an image. For instance, in the linked work by CodeParade, each song is represented as a piano roll, which when divided into measures of 96 beats each with 96 possible notes essentially becomes a visual repesentation of the song. Now instead of a temporal sequence of notes, each song is represented as a set of 16 sequential 96 x 96 images. Using my limited understanding of music theory, this representation could make sense for our NES music, since the music is very structured and repetitive where any given note depends more on the overal melody of the measures rather than the previous note or two. Below is a piano roll image of the first voice, P1, from an example track to illustrate this point.

<p align="center"><img src="/xstatic/example_piano_roll.png" width=1000></p>

Leveraging this framework, we decided to create our VAE models using the Keras library, implemented uing a Tensorflow backend in a python environment same as with the LSTM models. However, Keras does not include pre-built VAE models/layers, so we decided to construct our encoder and decoder sections using a deep network of fully connected layers. In some layers, we apply the same network recursively over each of the measures of our soundtracks to try and caprture the sequential nature of the data. Below is a more detailed list of the models layers:

1) Stage 1 - Encoder: Compresses inputs down to latent space of 2 vectors for sampling means and stds. Typically used latent dimension of ~100.
    - Layer 1: Time Distributed (over each measure of input track) Dense layer of about ~2000 nodes nodes with ReLU activations
    - Layer 2: Time Distributed Dense layer of about ~200 nodes with ReLU activations
    - Layer 3: Dense layer of about ~1600 nodes with ReLU activations which removes temporal dimension along measures
    - Layer 4: 2 Independent Dense layers of about ~100 nodes (our latent dim size) with identity activations
2) Stage 2 - Reparameterizer: Samples from normal distributions over latent variables to produce our final compressed representation.
3) Stage 3 - Decoder: Decompresses our latent representation back to original shape attemping to reproduce the original input as closely as possible.
    - Layer 1: Dense layer of about ~1600 nodes with ReLU activationsTime Distributed (over each measure of input track) Dense layer of about ~2000 nodes nodes with ReLU activations
    - Layer 2: Dense layer of about ~1600 nodes with ReLU activations which also reshapes data into temporal measures
    - Layer 3: Time Distributed Dense layer of about ~2000 nodes with ReLU activations
    - Layer 4: 1-4 Independent Dense layers of about ~10000 nodes (size of each meaure image) with sigmoid activations which rebuilds each of our input features

## Training

Both our LSTM and VAE models, reduced and full models, were trained in similar manners using the built in Model.fit method intrinsic to every Keras model. For training each model we used the seprsco tracks from NESMDB which was comprised of a training dataset of 4,502 NES soundtracks and a validation dataset of 403 NES soundtracks. However, recall that the inputs for each model were different, so the training and validation sets were also slightly different between the models, despite being derived from the same set of tracks. 

Since our LSTM models could take inputs of variable length, we decided to combine all the tracks for each dataset into a single long track from which to sample batches from. This means our training dataset was essentially a single track of ~3.5M notes from our 4,502 training songs, and likewise the validation set was a single track of ~268K notes from our 403 validation songs. Additionally, when combining these tracks we added our own unique note value between tracks to indicate the start/end of each song; this was done to help our model learn how songs start and end. Finally, when training the model we sampled sequences from these long tracks to create our training and validation batches. Typically we used sequence lengths in the 100-300 range since our PC/model could not handle longer sequences in memory and any thing shorter negatively affected the quality of the training.

As mentioned earlier, our VAE models take inputs with a well defined shape and length. As such, in creating the training/validation datasets for this model we had to parse each song for valid segments with the given length. Using samples with 8 measures comprised of 96 notes each (for a total length of 768 notes which is right at the average track length ~800 notes), this yielded 2766 valid samples for our training dataset and 190 for our validation dataset. Finally, when training the model we would sample our batches from each of thes datasets. For the full model, we also contructed generators for each dataset that would help produce the images for each batch, which take up a lot of memory, without storing the whole dataset in memory.

The optimized hyperparameters for our training perscriptions were worked out during experimentation primarily with the reduced models, but also to some degree with the full models. In general, we trained our models using batches of 100-300 samples per epoch and a complete training session was anywhere between 2,000 and 5,000 epochs. We found that using the built in Keras Adam optimizer worked best for adjusting the weights of our models, typically using an initial learning rate of .001 which exponentially decayed over the course of the training epochs. And lastly, for our LSTM models we used Kera's built in sparse categorical loss function, while for the VAE models we contructed unique loss functions that were a combination of a mean squared loss (measuring the difference between the input and output) and redularization term that forced our latent variables to be normally distributed. 

## Soundtrack Generation

Once our models were designed and trained, we finally moved on to the fun part of the project, using the models to actually create our own NES music. For each model type we had to handle the specifics of the song generation differently, since each model took different inputs/outputs, but in general the overall process was the same, as shown below:

1) Step 1: Reinitiate our trained model
2) Step 2: Create a starting point for the songs we will generate
        - LSTM: use the unique start/end identifier or a chosen segment of a song to be the starting point for each track
        - VAE: sample from our latent space or set the values for each latent variable to be used as the compressed representation of each track we want to generate
3) Step 3: Feed the starting points into our models to generate new NES tracks
        - LSTM: iteratively feed each note into the model and take the output to be the next note in the generated track
        - VAE: decode each of the latent vectors to generate/decompress each track
4) Step 4: Use custom built functions to extract valid NESM seprsco formatted tracks form what we generated
5) Step 5 (Optional): Use functions from the nesmdb library to convert the generated seprsco tracks to WAV format for listening to or to VGM for playing on NES synthasizers.

For the LSTM models, we essentially generate a single output track using an initial seed note(s) as a starting point. To do this, the models iteratively takes the current note in the sequence as an input, and predicts with some probabilty what the next note in the sequence should be. We then sample from this output distribution to choose our next note/input, and repeat the process for a set track length or until a set number of tracks are generated (which means our model outputs that number of start/end identifiers). Finally, we extract the valid songs from this single track by identifying the segments between start/end identifiers and reformatting the segments to valid serpsco, WAV, or VGM files.

In comparison, generating new tracks using the VAE is much simpler since the output of these models is a valid track with a fixed length. To generate a song using these models, we simply choose or randomly sample a compressed representation from our latent space and then decompress it to a full length track using just the decoder stage of the model. Personally, I prefer generating tracks using these models because the process is much simpler and reproduceable. The generative process with the LSTMs is essentially random, whereas with the VAEs I can store the latent vector used to make a song I liked and even adjust the latent values to fine-tune the song.

## Results

Using the above models and training perscriptions we were succesful in generating valid NES music. However, the quality of the music we were able to generate can only be classified as complete trash. The music was not very melodic and in most cases sounded like random noice across the combined 4 voices, especially for the LSTM models. Unfortunately, while the VAE models showed more promise on the reduced testing, we were not able to create a satisfactory working model for the full case. My PC just could not handle the size of the datasets when transformed to the image-like representation or allocate the OOM memory necessary for a decent sized model. 

We expected the results from our LSTM model to be sub-par since we found numerous examples of people attempting and failing at similar tasks using these models. As mentioned previously, it turns out that LSTM models are ill-suited for music generative tasks despite the fact that they have success with temporal sequence modeling. They tend to only have success when the music being modeled is simple, typically involving one instrument or voice, and free-form, which means the overall song has less structure, but the next note in the sequence is much more easily determined from the previous note. There are many theories behind this, but I personally side with the theory that human brains don't process and create music in a temporal manner. Despite the fact that music is played/listened to sequentially, our brains actually process the music in much larger chunks. We tend to process the song in chunks, and foucs more so on the overall pattern, melody, beat, chord progression, etc. of the song rather than each individual note. A slightly related topic to this is how the human brain processes words when reading. Despite the fact that writing follows a strict sequencial pattern, left to right and top to bottom, examples like the picture below show we don't actually read in this manner.

<p align="center"><img src="/xstatic/reading_example.jpeg" width=400></p>

Following the above reasoning, we hoped the VAE models would be much better suited for generating our music. Unfortunately, we couldn't get the full model to run, so the best songs we generated turned out to be from our LSTM model. The primary reason our VAE models failed was the increased size of the training data for our models. By reformatting the songs as piano roll image representations we effectively increased the size of our samples by 100 times, and it meant our full dataset was roughly comprised of 100K 100x100 pixel images or roughly 1B values. We were able to devise generators that avoided storing this dataset in memory, but we were unable to avoid the immense computational costs of transforming these samples using our model. My PC just did not have the working memory to transform these massive inputs through a realistically sized model. This is a shame since our reduced VAE model was very succesful at reproducing the input single melodic voices as shown in the below example.

<p align="center"><img src="/xstatic/original_track.png" width=1000></p>
<p align="center"><img src="/xstatic/reconstructed_track.png" width=1000></p>

In the future we would aim at addressing this issue first and foremost before we could expect to begin generating quality music. In the end, this may just come down to hardware and a more powerful CPU/GPU could solve the problem entirely. Furthermore, there are additional improvements that can be made to the datasets, the models themselves, and the training process to improve our music. While we put a lot of work into properly formatting the NES music soundtracks, we did little analysis of the musical structure of our songs. In the future it may prove valuable to analyse each song, identify certain structures, and pick the training segments from each accordingly, but this was seen as too much work at the moment. Furthermore, we believe it could be worth testing the use of CNNs, convolutional neural networks, in our VAE models moving forward. These types of models are incredibly succesful at abstracting patterns from images, so they may prove effective at encoding our piano roll representations. Furthermore, CNNs are less computationally intensive than a dense fully connected neural layer, so they may also address the failure of our full VAE model.

A few example WAV files produced by each of our models is included in the "NES_examples" folder to illustrate the resultant performance for each of our models. Also included in this folder is an original NES soundtrack from Zelda 2 to compare against.

## Acknowledgements 

There are two primary sources I would like to acknowledge for assisting me in this project. The first is Chris Donahue and [The NES Music Database](https://github.com/chrisdonahue/nesmdb) which provided the datasets used to train and validate my models. Additionally, his github has a python 2 library, called nesmdb, for manipulating and reformatting the files in the dataset which I edited to work in my python 3 environment. Secondly, I would like to acknowledge the youtube channel [CodeParade](https://www.youtube.com/channel/UCrv269YwJzuZL3dH5PCgxUw) and more specifically [his Neural Composer video](https://www.youtube.com/watch?v=UWxfnNXlVy8) which inspired me to try the same VAE technique for my own video game music generation. 
