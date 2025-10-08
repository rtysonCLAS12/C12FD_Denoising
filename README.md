# C12FD_Denoising
Code for the training and deployment of denoising in CLAS12's Forward Detector Drift Chambers

## Requirements

All required libraries are detailed in env_requirements.txt. To install using venvs and pip in a new environment called newenvname (you should choose a better name):

      python3 -m venv /path/to/env/location/newenvname
      source /path/to/env/location/newenvname/bin/activate.csh
      pip install torch torchvision 
      pip install lightning
      pip install matplotlib

Remember to always activate your environment before running code with source /path/to/env/location/newenvname/bin/activate.csh .

A script is provided to launch an interactive gpu session on ifarm. This is run with:

      source gpu_interactive
      module use /cvmfs/oasis.opensciencegrid.org/jlab/scicomp/sw/el9/modulefiles
      module load cuda/12.4.1

Note that in that case, we need to create a GPU friendly PyTorch environment. We can replace the earlier line with:
      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

## To Produce Training Data

I am using j4shell code to run the DenoiseExtractor.java code. This extracts all hits from the DC::tdc bank as our noisy input, and all hits from the TimeBasedTrkg::TBHits bank as our denoised output. The training data will be written in .csv files. The two sets of hits are written in 2D arrays filled with 1 for a hit and 0 otherwise, where the x axis is the number of wires in each drift chamber layers (112) and the y axis is the number of layers in the drift chamber (6 layer x 6 superlayers). The .csv file is written such that the noisy and denoised arrays are written one after the other (eg each event has 2 arrays consecutive arrays with noisy and denoised arrays). Arrays are written per sector, and only if TimeBasedTrkg::TBHits contains hits in that sector.

## To Train

In the new environment do:

      python3 train.py

This will launch a training script that uses the base model definition in trainer.py. The model is defined in the LitConvAutoencoder class which uses pytorch and pytorch-lightning for the model definition. A few things to note. First, we are only pooling after the first two encoding layers, and upsampling in the last two decoding layers. We also apply some dropout layers in the first encoding block to avoid overfitting. This adds some stochasticity to the training procedure which avoids overfitting but means the model doesn't always predict the same output for the same input (the difference should be very small).

The trained network will be saved in the nets/ directory. The train script also produces some plots, saved by default in the plots/ directory. The test.py script can be used to make the plots without retraining. The plotter class contains the plotting function definitions. It takes as arguments when initialised the location where plots are written (eg plots/), an "end_name" string to append to plot names so that they are not overwritten, and the sector number.

The many_train.py script allows to sweep through a user-defined grid of model hyperparameters. This allows to test how certain hyperparameters affect the performance of the model. At the moment, the tested hyperparameters contain the number of filters in each convolutional layer, the number of convolutional blocks, and the kernel size for each convolutional layer.

The Data.py file contains two functions, one to load the data from the saved .csv files (note that it has hardcoded data shapes) and the second functions computes the signal efficiency and background rejection for a range of thresholds.


## To Deploy

The repository includes a toy maven project that allows to load the trained networks using the Deep Java Libray and applies them to a toy input. The pom.xml file contains the required set-up (assumes jdk/23.0.1) and the src/main/java/org/example/Main.java class contains a class capable of loading and applying the model. The class contains two main parts, first a translator that allows to convert a 2D array of floats into an input suitable for the network and convert the output of the network into a 2D array of floats. Second, the a Criteria class defines the path to the model and the engine (in this case pytorch).




