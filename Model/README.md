# HOW TO RUN THE MODEL LOCALLY:

## Setting up virtual environment 

1. Make sure you have a virtual environment that is suitable for it. Run these commands in the terminal:

```shell
#Install Python 3.10.13 via pyenv if you haven't already
pyenv install 3.10.13

#Create a virtual environment (replace [YOUR_PATH] with your desired path)
python3.10 -m venv [YOUR_PATH]

#Activate the virtual environment
source [YOUR_PATH]/bin/activate
```

2. Download TensorFlow packages that allow for Apple Silicon to be used more effectively through MPS 
```shell
pip install tensorflow-macos tensorflow-metal
```

## Downloading the dataset for local training and fixing the pathing

3. You need to download the HAM10000 Skin Cancer Dataset from Kaggle (https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000?resource=download) 

- It will come as a zip. Once you unzip it, you will see a metadata CSV and two folders of the pictures that are labeled part 1 and part 2.
- You need to go into model.py and reroute the paths for the metadata CSV **and** the two parts data folders in accordance with where you have the dataset.
- Do the same thing for load_model.py 

## Running the scripts

4. Run model.py to train the model using that dataset. Should take a couple of minutes.

- python model.py

This script will save the trained weights to the root directory. 
- dermalite_mobilenet_model.h5

5. Run convert.py to generate the .mlpackage 
The conversion script takes the trained Keras H5 file and outputs the optimized Core ML format. Like all the other files, verify that the paths are correct with regards to your machine. 
- This will produce dermalite_model.mlpackage

6. To load/run the model, use the load_model.py script to check the accuracy.

- python load_model.py 