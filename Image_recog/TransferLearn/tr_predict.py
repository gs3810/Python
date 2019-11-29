###https://appliedmachinelearning.blog/2019/07/29/transfer-learning-using-feature-extraction-from-trained-models-food-images-classification/
import numpy as np
from keras.models import load_model
from keras.applications import VGG16     #test resnet if possible
from keras.applications import imagenet_utils
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import warnings
warnings.filterwarnings('ignore')

def create_features(dataset, pre_model):
    x_scratch = []
    try:
        for imagePath in dataset:
            # load the input image and image is resized to 224x224 pixels
            image = load_img(imagePath, target_size=(224, 224))
            image = img_to_array(image)
            # preprocess the image 
            image = np.expand_dims(image, axis=0)
            image = imagenet_utils.preprocess_input(image)
            # add the image to the batch
            x_scratch.append(image)
    except:
        pass
    x = np.vstack(x_scratch)
    features = pre_model.predict(x, batch_size=32)
    features_flatten = features.reshape((features.shape[0], 7 * 7 * 512))
    return x, features, features_flatten

def predict(path, model,model_transfer):
    new_test = list(['Food-11/test/kinley1.jpg'])
    new_test_x, new_test_features, new_test_features_flatten = create_features(new_test, model)
    prob = model_transfer.predict(new_test_features)
    return prob

num_classes = 4
classes = ['kinley','knuckles','ozone','spark'] 
 
# load VGG feature weights
model = VGG16(weights="imagenet", include_top=False)
model.summary()

model_transfer = load_model("model/tr_model.h5")

prob = predict('Food-11/test/kinley1.jpg',model,model_transfer)
if np.max(prob)>0.8:
    pass
else:
    pred = ['unknown']     
print (np.round(prob,2))








