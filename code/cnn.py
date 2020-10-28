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

# CNN Module
# Convolutional neural network model.

import os

import matplotlib.pyplot as plt
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.models import Sequential

plt.rcParams.update({"font.size": 12})


class CNN:

  def __init__(self, height, width, depth, class_count):
    self._height = height
    self._width = width
    self._depth = depth
    self._class_count = class_count

    self._model = self._build_model()

  def _build_model(self):
    input_shape = (self._height, self._width, self._depth)

    # Build the model and compile it.
    model = Sequential()

    model.add(Conv2D(128, (5, 5), input_shape=input_shape, padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (5, 5), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(Dropout(0.2))

    model.add(Dense(self._class_count, activation="softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model

  def _save_training_plots(self, directory_path, training_history):
    if not os.path.isdir(directory_path):
      print("Output directory does not exist | Creating directories along directory path")
      os.makedirs(directory_path)

    print("Saving training history | Directory path: {}".format(directory_path))

    plt.plot(training_history["acc"])
    if "val_acc" in training_history.keys():
      plt.plot(training_history["val_acc"])
    plt.title("CNN accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    if "val_acc" in training_history.keys():
      plt.legend(["Training set", "Validation set"], loc="upper left")
    plt.savefig("{}/cnn-training-accuracy".format(directory_path))
    plt.close()

    plt.plot(training_history["loss"])
    if "val_loss" in training_history.keys():
      plt.plot(training_history["val_loss"])
    plt.title("CNN loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    if "val_loss" in training_history.keys():
      plt.legend(["Training set", "Validation set"], loc="upper left")
    plt.savefig("{}/cnn-training-loss".format(directory_path))
    plt.close()

  def _save_test_results(self, directory_path, test_results):
    if not os.path.isdir(directory_path):
      print("Output directory does not exist | Creating directories along directory path")
      os.makedirs(directory_path)

    print("Saving test_results | Directory path: {}".format(directory_path))

    with open("{}/cnn-test-results.txt".format(directory_path), "w+") as output_file:
      output_file.write("Model's loss on unseen data: " + str(test_results[0]))
      output_file.write("\nModel's accuracy on unseen data: " + str(test_results[1]))

  def train(self, x_train, y_train, epochs, batch_size, validation_split, output_path):
    training = self._model.fit(x_train, y_train, validation_split=validation_split, epochs=epochs,
                               batch_size=batch_size)
    self.save_model(output_path)
    self._save_training_plots(output_path, training.history)

  def test(self, x_test, y_test, output_path):
    test_results = self._model.evaluate(x_test, y_test)
    self._save_test_results(output_path, test_results)

  def save_model(self, directory_path):
    if not os.path.isdir(directory_path):
      print("Output directory does not exist | Creating directories along directory path")
      os.makedirs(directory_path)

    print("Saving model | Directory path: {}".format(directory_path))

    self._model.save("{}/cnn.h5".format(directory_path))

  def summary(self):
    self._model.summary()
