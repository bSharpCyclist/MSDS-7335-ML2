#***********************
# Title: HW4
# Purpose: Transfer Learning
# Author: Dan Crouthamel, Fabio Savorgnan
# Date: April 2021
#***********************
#
# Proposed Solution:
# We tried several attempts: 
# 1. At first using EMNIST data, it works very nicely. Very high accuracy, but lazy because we didn't draw images
# 2. Next we tried drawing on paper with lines and got an accuracy of around 72%. The lines killed us I think.
# 3. Third attempt goes like this.
# In Microsoft Word, create a grid of 10 rows by 5 columns
# Then with a touch screen monitor or laptop, draw letters in each cell
# Then remove grid lines and take a screenshot such that the width is divisible by 5 and height by 10
# For example, our LetterSheetTrain.png has dimensions of Width = 870 pixels, Height = 1190 pixels
# We will then use python code to create 50 images from the one single image
# We will invert and test again. I suspect the accuracy should be very high
# We need a test/validation sheet as well. This has 25 images, LetterSheetTest.png, 870x590.
# We will break LetterSheetTest up into 25 images.


import numpy as np
import cv2

# Read in our Training Letters
img = cv2.imread(('./data/LetterSheetTrain.png'), 0)
img = np.asarray(img)

# Show entire sheet of images
cv2.imshow('image',img)
cv2.waitKey(0)

M = img.shape[0]//10
N = img.shape[1]//5

# Create a list of training images, the length when printed out should be 50
# Stole list comprehension code from here
# https://stackoverflow.com/questions/5953373/how-to-split-image-into-multiple-pieces-in-python
train_letters = [img[x:x+M,y:y+N] for x in range(0,img.shape[0],M) for y in range(0,img.shape[1],N)]
print(len(train_letters))

# Show a random training image from the set of 50, make sure it looks good
cv2.imshow('image',train_letters[17])
cv2.waitKey(0)

# Now do the same for our testing images, read in a different sheet
img = cv2.imread(('./data/LetterSheetTest.png'), 0)
img = np.asarray(img)

# In this case we have a 5x5 grid
M = img.shape[0]//5
N = img.shape[1]//5

# Create a list of testing images, the length when printed out should be 25
test_letters = [img[x:x+M,y:y+N] for x in range(0,img.shape[0],M) for y in range(0,img.shape[1],N)]
print(len(test_letters))

# Show a random testing image from the set of 25, make sure it looks good
cv2.imshow('image',test_letters[8])
cv2.waitKey(0)

# Now Bring in Stuart's code

from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, Model

# Define function to plot training curves
def plot_training_curves(history, title=None):
    ''' Plot the training curves for loss and accuracy given a model history
    '''
    # find the minimum loss epoch
    minimum = np.min(history.history['val_loss'])
    min_loc = np.where(minimum == history.history['val_loss'])[0]
    # get the vline y-min and y-max
    loss_min, loss_max = (min(history.history['val_loss'] + history.history['loss']),
                          max(history.history['val_loss'] + history.history['loss']))
    acc_min, acc_max = (min(history.history['val_accuracy'] + history.history['accuracy']),
                        max(history.history['val_accuracy'] + history.history['accuracy']))
    # create figure
    fig, ax = plt.subplots(ncols=2, figsize = (15,7))
    fig.suptitle(title)
    index = np.arange(1, len(history.history['accuracy']) + 1)
    # plot the loss and validation loss
    ax[0].plot(index, history.history['loss'], label = 'loss')
    ax[0].plot(index, history.history['val_loss'], label = 'val_loss')
    ax[0].vlines(min_loc + 1, loss_min, loss_max, label = 'min_loss_location')
    ax[0].set_title('Loss')
    ax[0].set_ylabel('Loss')
    ax[0].set_xlabel('Epochs')
    ax[0].legend()
    # plot the accuracy and validation accuracy
    ax[1].plot(index, history.history['accuracy'], label = 'accuracy')
    ax[1].plot(index, history.history['val_accuracy'], label = 'val_accuracy')
    ax[1].vlines(min_loc + 1, acc_min, acc_max, label = 'min_loss_location')
    ax[1].set_title('Accuracy')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_xlabel('Epochs')
    ax[1].legend()
    plt.show()

# Download MNIST DATA
(mnist_data, mnist_labels), (mnist_test_data, mnist_test_labels) = keras.datasets.mnist.load_data()
print(mnist_data.shape)
print(mnist_test_data.shape)

# Preprocess and Reshape MNIST Data
mnist_data = mnist_data / 255.
mnist_data = mnist_data.reshape(mnist_data.shape + (1,))

mnist_test_data = mnist_test_data / 255.
mnist_test_data = mnist_test_data.reshape(mnist_test_data.shape + (1,))

print(mnist_data.shape)
print(mnist_test_data.shape)

# Setup model to learn from MNIST digits, will transfer to letters later
num_classes=10
filters=32
pool_size=2
kernel_size=3
dropout=0.2
input_shape = (28,28,1)

model = Sequential([
      # convolutional feature extraction
      # ConvNet 1
      keras.layers.Conv2D(filters, kernel_size, padding = 'valid',
              activation='relu',
              input_shape=input_shape),
      keras.layers.MaxPooling2D(pool_size=pool_size),

      # ConvNet 2
      keras.layers.Conv2D(filters, kernel_size,
              padding = 'valid',
              activation='relu'),
      keras.layers.MaxPooling2D(pool_size=pool_size),

      # classification 
      # will retrain from here
      keras.layers.Flatten(name='flatten'),

      keras.layers.Dropout(dropout),
      keras.layers.Dense(128, activation='relu'),
      
      keras.layers.Dropout(dropout, name='penult'),
      keras.layers.Dense(num_classes, activation='softmax', name='last')
  ])

es = keras.callbacks.EarlyStopping(min_delta=0.001, patience=2)

model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='adam', #sgd, nadam, adam, rmsprop
                      metrics=['accuracy'])

history = model.fit(mnist_data, mnist_labels,
                    validation_data=(mnist_test_data, mnist_test_labels),
                    batch_size=32,
                    epochs=1000,
                    callbacks=[es])

# Plot the Training Curves
plot_training_curves(history=history)

# Print out the Model Summary
model.summary()

# Lock the ConvNet Layers
layer_trainable = False
for layer in model.layers:
  layer.trainable = layer_trainable

  if layer.name == 'flatten':
    layer_trainable = True

print(f"{'Layer Name':17} {'Is Trainable?'}")
for layer in model.layers:
  print(f"{layer.name:17} {layer.trainable}")

# get the penultimate layer of the model
penult_layer = model.get_layer(name='penult')

# create a new output layer
output_layer = keras.layers.Dense(5, activation='softmax')(penult_layer.output)
new_model = Model(model.input, output_layer)
new_model.summary()


# Now Setup Our Training Images
# We will invert the image, scale to between 0 and 1 and resize to 28x28
new_images = []
width = 28
height = 28
dim = (width, height)

for i in range(len(train_letters)):
    img = train_letters[i]
    img = cv2.bitwise_not(img)
    img = img / 255.
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    new_images.append(np.array(resized))

x_letters_train = np.stack(new_images)
print(x_letters_train.shape)

x_letters_train = x_letters_train.reshape((50,28,28,1))
print(x_letters_train.shape)

# Setup our Y labels for Training
# Note, the way the code splits images is row by row
# So it goes A, B, C, D, E, and then A, B .. etc.
y_labels = [0, 1, 2, 3, 4] * 10
y_labels = np.asarray(y_labels)

# Now Setup Our Test/Validation Images
validation_new_images = []

for i in range(len(test_letters)):
    img = test_letters[i]
    img = cv2.bitwise_not(img)
    img = img / 255.
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    validation_new_images.append(np.array(resized))

validation_letter_data = np.stack(validation_new_images)
print(validation_letter_data.shape)

validation_letter_data = validation_letter_data.reshape((25,28,28,1))
print(validation_letter_data.shape)

# Setup our Y labels for testing
y_validation_labels = [0, 1, 2, 3, 4] * 5
y_validation_labels = np.asarray(y_validation_labels)

new_model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

new_model_hist = new_model.fit(x_letters_train, y_labels,
                              validation_data=(validation_letter_data, y_validation_labels),
                              batch_size=32,
                              epochs=1000,
                              callbacks=[es])


plot_training_curves(new_model_hist)