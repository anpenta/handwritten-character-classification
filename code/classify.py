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

# Classify Module
# Module to classify character images using a trained CNN.

from keras.models import load_model

import utility

input_arguments = utility.parse_input_arguments(module="classify")

# Load and preprocess the images.
images, filenames = utility.load_images(input_arguments.image_path)
images = utility.preprocess(images)

# Load the model and make the predictions.
cnn = load_model(input_arguments.cnn_path)
predictions = cnn.predict(images)

analyzed_predictions = utility.analyze(predictions, filenames)
utility.save_dataframe(analyzed_predictions, input_arguments.output_path, "analyzed_predictions")
