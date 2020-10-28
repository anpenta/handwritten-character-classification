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

# Train Module
# Module to train a CNN model on character data.

import numpy as np

import cnn
import utility

input_arguments = utility.parse_input_arguments(module="train")
images, labels = utility.load_data(input_arguments.image_path)
class_count = len(np.unique(labels))

images, labels = utility.preprocess(images, labels)
images, labels = utility.shuffle(images, labels)
x_train, x_test, y_train, y_test = utility.split(images, labels, test_size=0.2)

cnn = cnn.CNN(x_train.shape[1], x_train.shape[2], x_train.shape[3], class_count)
cnn.summary()

cnn.train(x_train, y_train, epochs=input_arguments.epochs, batch_size=input_arguments.batch_size,
          validation_split=input_arguments.validation_split, output_path=input_arguments.output_path)
cnn.test(x_test, y_test, output_path=input_arguments.output_path)
