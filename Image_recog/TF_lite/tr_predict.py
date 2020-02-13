#https://medium.com/towards-artificial-intelligence/testing-tensorflow-lite-image-classification-model-e9c0100d8de3 
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils

saved_model_dir = 'models/gab_model'
model = tf.keras.models.load_model('models/gab_model',compile=True) 

imagePath = 'test/osamp/test_kinley.jpg'
image = load_img(imagePath, target_size=(224, 224))
image = img_to_array(image)/255  # preprocess to fit NN network
image = np.expand_dims(image, axis=0)

# include read directories in order
tf_pred = np.round(model.predict(image),2)


