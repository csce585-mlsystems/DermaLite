# HOW TO RUN THE MODEL LOCALLY:

# Setting up virtual environment 

1. Make sure you have an virtual environment that is suitable for it. Run these commands on terminal:

Install Python 3.10.13 via pyenv if you haven't already
- pyenv install 3.10.13

Create a virtual environment (replace [YOUR_PATH] with your desired path)
- python3.10 -m venv [YOUR_PATH]

Activate the virtual environment
- source [YOUR_PATH]/bin/activate


2. Download TensorFlow packages that allows for Apple Silicon to be used more effectively through MPS 

- pip install tensorflow-macos tensorflow-metal

# Downloading the dataset for local training and fixing the pathing

3. You need to download the HAM10000 Skin Cancer Dataset from Kaggle (https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000?resource=download) 

- It will come as a zip. Once you unzip it, you will see a metadata csv and two folders of the pictures that is labeled part 1 and part 2. You need to go into model.py and reroute the path for metadata csv and the two paths in accordance to where you have the dataset. Do the same thing for load_model.py 

# Running the scripts

4. Run model.py to train the model using that dataset. Should take a couple of minutes.

- python model.py

53. To load/run the model, use the load_model.py script 

- python load_model.py 