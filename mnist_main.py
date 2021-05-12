# Dependencies to Visualize the model

from IPython.display import Image, SVG
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(0)


# Filepaths, numpy, and Tensorflow
import os
print("PYTHONPATH:", os.environ.get('PYTHONPATH'))
print("PATH:", os.environ.get('PATH'))
import tensorflow as tf


# Sklearn scaling
from sklearn.preprocessing import MinMaxScaler


# Keras Specific Dependencies
# Keras
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import fashion_mnist


# Loading and Preprocessing our Data
# Load the Fashion MNIST Dataset from Keras

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

## Explore the data

print(train_images.shape)

print(len(train_labels))

print(test_images.shape)

print(len(test_labels))

# Preprocess the data

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()


# Each Image is a 28x28 Pixel greyscale image with values from 0 to 255
train_images = train_images / 255.0

test_images = test_images / 255.0

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()


# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
]).add(Dense(32, activation='softmax', input_shape=(10,)))

## Compile the model

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the MODEL

## Feed the model

print(model.fit(train_images, train_labels, epochs=10))

# Saving and Loading models
# We can save our trained models using the HDF5 binary format with the extension `.h5`

# Save the model
model.save("mnist_trained.h5")

from tensorflow.keras.models import load_model
model = load_model("mnist_trained.h5")

# Evaluating the Model

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

# Make predictions
# We can use our trained model to make predictions using model.predict

probability_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_images)

print(predictions[0])

np.argmax(predictions[0])

print(test_labels[0])

## Graph this to look at the full set of 10 class predictions

def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

## Verify predictions

i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

i = 5
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()


# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

## Use the trained model

# Import a Custom Image
filepath = "Images/tshirt.png"

from tensorflow.keras.preprocessing import image
image_size = (28, 28)
im = image.load_img(filepath, target_size=image_size, color_mode="grayscale")
print(im)

# Convert the image to a numpy array 
from tensorflow.keras.preprocessing.image import img_to_array
image = img_to_array(im)
print(image.shape)

# Scale the image pixels by 255 (or use a scaler from sklearn here)
image /= 255

# Flatten into a 1x28*28 array 
img = image.flatten().reshape(-1, 28*28)
print(img.shape)

plt.imshow(img.reshape(28, 28), cmap=plt.cm.Greys)

img = 1 - img
plt.imshow(img.reshape(28, 28), cmap=plt.cm.Greys)

# Make predictions

predictions_single = probability_model.predict(img)

print(predictions_single)

predictions_percentage = np.round(predictions_single, 3)*100
print(predictions_percentage)

res = probability_model.predict(img)
results = [[i,r] for i,r in enumerate(res)]
results.sort(key=lambda x: x[1], reverse=True)
for r in results:
    print(class_names, str(r[1]))
    
print(predictions_percentage)
print(results)
print(predictions_single)
print(class_names)

# Create our figure and data we'll use for plotting
fig, ax = plt.subplots(figsize=(9, 3))

color=['red']*len(predictions_percentage[0])

np_predictions_percentage = np.array(predictions_percentage[0])
max_index_col = np.argmax(np_predictions_percentage, axis=0)

color[max_index_col]='blue'

ax.bar(class_names, predictions_percentage[0], color=color)

ax.grid(False)
ax.set_xticks(range(10))
ax.set_yticks([])

ax.set_ylim([0, 100])
predicted_label = np.argmax(predictions_percentage[0])

for j, p in enumerate(predictions_percentage[0]):
    if p>1:
        ax.text(j-0.2, p+2, str(p), color='blue', fontweight='bold')