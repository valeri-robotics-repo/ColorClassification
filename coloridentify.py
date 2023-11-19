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

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

data_dir = pathlib.Path("C:/Datasets/sorted")
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)
img_height = 100
img_width = 100

batch_size = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.1,
  shuffle=True,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  shuffle=True,
  validation_split=0.1,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

num_classes = 12


model = create_model(img_height, img_width, num_classes)
model.summary()
epochs=20
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)


# Save the weights
model.save_weights('./saved_model')

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

