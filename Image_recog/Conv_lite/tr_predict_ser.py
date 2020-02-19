#https://medium.com/towards-artificial-intelligence/testing-tensorflow-lite-image-classification-model-e9c0100d8de3 
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
import glob
import imageio

def read_image(imagePath):

    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image)/255  # preprocess to fit NN network
    image = np.expand_dims(image, axis=0)
    return image

saved_model_dir = 'model'
model = tf.keras.models.load_model(saved_model_dir,compile=True) 

# read in labels
labels = pd.read_excel('labels/Labels.xlsx').iloc[:,1:]

tf_out =[]
for file in glob.glob('test/*.jpg'):
    image = read_image(file)
    tf_pred = pd.DataFrame(model.predict(image), columns=labels.columns).round(2)
    tf_out.append(tf_pred)
    
# pd.concat[] convert to df and out to excel
