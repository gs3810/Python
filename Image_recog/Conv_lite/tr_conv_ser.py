"""https://www.tensorflow.org/lite/guide/inference"""
import tensorflow as tf
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np
import pandas as pd

# convert to tf.lite model
saved_model_dir = 'models/gab4sig_model'
TFLITE_MODEL = 'models/garb_recog4.tflite' 

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()
open(TFLITE_MODEL, "wb").write(tflite_model)

tflite_interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL)
tflite_interpreter.allocate_tensors()

input_details = tflite_interpreter.get_input_details()
output_details = tflite_interpreter.get_output_details()

# Test model on image
imagePath = 'test/osamp/test_metal_1.jpg'
image = load_img(imagePath, target_size=(224, 224))
image = img_to_array(image)/255  # preprocess to fit NN network
image = np.expand_dims(image, axis=0)

tflite_interpreter.set_tensor(input_details[0]['index'], image)        # Tensor is symbolic handle to one of the outputs of an Operation. It does not hold the values of that operation's output, but instead provides a means of computing those values in a TensorFlow tf.compat.v1.Session.

# Invoke the interpreter
tflite_interpreter.invoke()

# labels
labels = pd.read_excel("models/Labels.xlsx").iloc[:,1:]

# get the output data
output_data = pd.DataFrame(tflite_interpreter.get_tensor(output_details[0]['index']),columns=labels.columns).round(2) 
print(output_data)



