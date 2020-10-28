# Handwritten Character Classification Deep Learning

This repository contains a deep learning system that performs handwritten character classification on ancient Hebrew character images using a convolutional neural network (CNN). Unfortunately the dataset was provided confidentially and cannot be published. However, the system can be easily adapted to perform classification with different classes.

## Installation

It is recommended to install conda and then create an environment for the system using the ```environment.yaml``` file. A suggestion on how to install the system and activate the environment is provided below.

```bash
git clone https://github.com/anpenta/handwritten-character-classification-deep-learning.git
cd handwritten-character-classification-deep-learning
conda env create -f environment.yaml
conda activate handwritten-character-classification-deep-learning
```

## Running the system

To run the system for training you can provide commands through the terminal using the ```train``` module. An example is given below.

```bash
python3 train.py ./character-data 50 8 0.2 ./output
```
This will train a model using data from the ```./character-data``` directory for 50 epochs with a batch size of 8 and a validation split of 0.2 and save the training plots and the trained model in the ```./output``` directory. An example of how to see the parameters for training is provided below.

```bash
python3 train.py --help
```

To run the system for classification you can provide commands through the terminal using the ```classify``` module. An example is given below.


```bash
python3 classify.py ./character-data ./cnn.h5 ./output
```
This will classify the images from the ```./character-data``` directory using the CNN from the ```./cnn.h5``` file and save the predictions in the ```./output``` directory. An example of how to see the parameters for classification is provided below.

```bash
python3 classify.py --help
```

## Results

As an example, below are the training results we get after training a model for 50 epochs with a batch size of 32 and a validation split of 0.2. The model starts overfitting after about ten training epochs.

<p float="left">
<img src=./training-results/cnn-training-accuracy.png height="320" width="420">
<img src=./training-results/cnn-training-loss.png height="320" width="420">
</p>

## Sources
* LeCun, Yann, et al. "Object recognition with gradient-based learning." Shape, contour and grouping in computer vision. Springer, Berlin, Heidelberg, 1999. 319-345.
