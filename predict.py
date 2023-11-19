import matplotlib.pyplot as plt
import numpy as np
import os
import PIL

import tensorflow as tf
#print(tf.__version__)
#print(tf.test.is_gpu_available())

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from create_model import *
import glob

import pathlib

img_height = 100
img_width = 100

num_classes = 12


class_names = ['beige', 'black', 'blue', 'brown', 'green', 'orange', 'pink', 'red', 'silver', 'white', 'yellow']

model = create_model(img_height, img_width, num_classes)
model.summary()

# Restore the weights
model.load_weights('./saved_model')

# Evaluate the model
#loss, acc = model.evaluate(test_images, test_labels, verbose=2)
#print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

img = keras.preprocessing.image.load_img(
    "C:/Datasets/sorted/beige/yellow2_bowl2.jpg", target_size=(img_height, img_width)
)

img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

