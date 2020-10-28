# Copyright (C) 2020 Andreas Pentaliotis
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

# Utility Module
# Utility functions to run handwritten character classification.

import cv2 as cv
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


def plot(name, image):
  cv.namedWindow(name, cv.WINDOW_NORMAL)
  cv.imshow(name, image)
  cv.waitKey(0)
  cv.destroyWindow(name)


def read_image(image_path):
  image = cv.imread(image_path, cv.IMREAD_UNCHANGED)
  return image


def load_data(directory_path):
  print("Loading data")
  
  # Load the images and labels.
  images = []
  labels = []
  for directory in os.listdir("{}/".format(directory_path)):
    for filename in os.listdir("{}/{}/".format(directory_path, directory)):
      image = read_image("{}/{}/{}".format(directory_path, directory, filename))
      images.append(image)
      labels.append(directory)

  return images, labels


def load_images(directory_path):
  print("Loading images")

  # Load the images and their filenames.
  images = []
  filenames = []
  for filename in os.listdir("{}/".format(directory_path)):
    image = read_image("{}/{}".format(directory_path, filename))
    images.append(image)
    filenames.append(filename)

  return images, filenames


def shuffle(images, labels):
  print("Shuffling data")

  assert len(images) == len(labels)
  permutation = np.random.permutation(len(images))

  return images[permutation], labels[permutation]


def split(images, labels, test_size):
  print("Splitting data")

  (x_train, x_test, y_train, y_test) = train_test_split(images, labels, test_size=test_size, random_state=1)
  
  return x_train, x_test, y_train, y_test


def resize(images, height, width):
  return [cv.resize(x, (height, width), interpolation=cv.INTER_CUBIC) for x in images]


def binarize(images):
  images = [cv.threshold(x, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1] for x in images]
  images = [x / 255 for x in images]
  return images


def preprocess(images, labels=None):
  print("Preprocessing data")

  # Preprocess the images.
  images = resize(images, height=64, width=64)
  images = binarize(images)
  images = np.array(images)
  images = np.reshape(images, (images.shape[0], images.shape[1], images.shape[2], 1))

  if labels is not None:
    # One hot encode the labels.
    binarizer = LabelBinarizer()
    labels = binarizer.fit_transform(labels)
    return images, labels

  return images


def analyze(predictions, filenames):
  print("Analyzing predictions")

  # Determine the labels and probabilities and store them in a dataframe with
  # the image filenames.
  analyzed_predictions = pd.DataFrame()
  labels = ["Alef", "Ayin", "Bet", "Dalet", "Gimel", "He", "Het", "Kaf", "Kaf-final",
            "Lamed", "Mem", "Mem-medial", "Nun-final", "Nun-medial", "Pe", "Pe-final",
            "Qof", "Resh", "Samekh", "Shin", "Taw", "Tet", "Tsadi-final", "Tsadi-medial",
            "Waw", "Yod", "Zayin"]
  indices = np.argmax(predictions, axis=1)
  probabilities = np.max(predictions, axis=1)
  analyzed_predictions["filename"] = filenames
  analyzed_predictions["label"] = [labels[x] for x in indices]
  analyzed_predictions["probability"] = probabilities

  return analyzed_predictions


def save_dataframe(dataframe, directory_path, basename):
  if not os.path.isdir(directory_path):
    print("Output directory does not exist | Creating directories along directory path")
    os.makedirs(directory_path)

  filename = basename + ".csv"
  print("Saving data | Filename: {} | Directory path: {}".format(filename, directory_path))
  dataframe.to_csv("{}/{}".format(directory_path, filename), index=False)


def parse_input_arguments(module, epoch_choices=range(1, 51, 1), batch_size_choices=range(8, 33, 8),
                          validation_split_choices=(0.0, 0.05, 0.1, 0.15, 0.2)):
  parser = None
  if module == "classify":
    parser = argparse.ArgumentParser(prog="classify", usage="classifies the provided images using the provided trained"
                                                            " CNN model")
    parser.add_argument("image_path", help="directory path to character image data")
    parser.add_argument("cnn_path", help="directory path to trained CNN model")
    parser.add_argument("output_path", help="directory path to save the predictions")
  elif module == "train":
    parser = argparse.ArgumentParser(prog="train", usage="trains a CNN model on the provided image data")
    parser.add_argument("image_path", help="directory path to training image data; data should be organized in"
                                           " subdirectories using their labels as subdirectory names")
    parser.add_argument("epochs", type=int, choices=epoch_choices, help="number of training epochs")
    parser.add_argument("batch_size", type=int, choices=batch_size_choices,
                        help="training data batch size")
    parser.add_argument("validation_split", type=float, choices=validation_split_choices,
                        help="percentage of training data to use as validation set")
    parser.add_argument("output_path", help="directory path to save the output of training")

  input_arguments = parser.parse_args()

  return input_arguments
