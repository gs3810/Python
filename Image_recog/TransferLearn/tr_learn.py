###https://appliedmachinelearning.blog/2019/07/29/transfer-learning-using-feature-extraction-from-trained-models-food-images-classification/
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt; 
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
#from hypopt import GridSearch
from keras.utils import np_utils
from keras.models import Sequential
from keras.applications import VGG16     #test resnet if possible
from keras.applications import imagenet_utils
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.layers import Dense, Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, GlobalAveragePooling2D
import warnings
warnings.filterwarnings('ignore')

def show_imgs(X):
    plt.figure(figsize=(8, 8))
    k = 0
    for i in range(0,4):
        for j in range(0,4):
            image = load_img(train[k], target_size=(224, 224))
            plt.subplot2grid((4,4),(i,j))
            plt.imshow(image)
            k = k+1
    # show the plot
    plt.show()

def create_features(dataset, pre_model):
    x_scratch = []
    try:
        # loop over the images
        for imagePath in dataset:
            # load the input image and image is resized to 224x224 pixels
            image = load_img(imagePath, target_size=(224, 224))
            image = img_to_array(image)
     
            # preprocess the image by (1) expanding the dims and (2) subtracting mean RGB pixel intensity from the ImageNet dataset
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

def create_model(train_features):
    #build final covolutional network
    model_transfer = Sequential()
    model_transfer.add(GlobalAveragePooling2D(input_shape=train_features.shape[1:]))
    model_transfer.add(Dropout(0.2))
    model_transfer.add(Dense(100, activation='relu'))
    model_transfer.add(Dense(num_classes, activation='softmax'))
    model_transfer.summary()
    return model_transfer

def predict(path, model):
    new_test = list(['Food-11/test/kinley1.jpg'])
    new_test_x, new_test_features, new_test_features_flatten = create_features(new_test, model)
    prob = model_transfer.predict(new_test_features)
    preds = str(le.inverse_transform(np.argmax(model_transfer.predict(new_test_features), axis=1)))
    return prob, preds

train = [os.path.join("Food-11/training/",img) for img in os.listdir("Food-11/training")]
val = [os.path.join("Food-11/validation/",img) for img in os.listdir("Food-11/validation")]
test = [os.path.join("Food-11/evaluation/",img) for img in os.listdir("Food-11/evaluation")]

print(len(train),len(val),len(test))

train_y = [img.split("/")[-1].split("_")[0]for img in train][:-1]
val_y = [img.split("/")[-1].split("_")[0]for img in val][:-1]
test_y = [img.split("/")[-1].split("_")[0]for img in test][:-1]
num_classes = 4 
 
# Convert class labels in one hot encoded vector
le = LabelEncoder()
y_train = le.fit_transform(train_y)
y_train = np_utils.to_categorical(y_train, num_classes)

y_val = le.transform(val_y)
y_val = np_utils.to_categorical(y_val, num_classes)

y_test = le.transform(test_y)
y_test = np_utils.to_categorical(y_test, num_classes)

print("training data available in classes; ",[train_y.count(i) for i in range(0,num_classes)])
 
food_classes = tuple(set(train_y))
 
y_pos = np.arange(len(food_classes))
counts = [train_y.count(i) for i in range(0,num_classes)]
 
plt.barh(y_pos, counts, align='center', alpha=0.5)
plt.yticks(y_pos, food_classes)
plt.xlabel('Counts')
plt.title('Train Data Class Distribution')
plt.show()

#show_imgs(train)
 
# chop the top dense layers, include_top=False
model = VGG16(weights="imagenet", include_top=False)
model.summary()

train_x, train_features, train_features_flatten = create_features(train, model)
val_x, val_features, val_features_flatten = create_features(val, model)
test_x, test_features, test_features_flatten = create_features(test, model)

print(train_x.shape, train_features.shape, train_features_flatten.shape)
print(val_x.shape, val_features.shape, val_features_flatten.shape)
print(test_x.shape, test_features.shape, test_features_flatten.shape)

# Creating a checkpointer
checkpointer = ModelCheckpoint(filepath='model/scratchmodel.best.hdf5',verbose=1,save_best_only=True)
    
model_transfer = create_model(train_features)
model_transfer.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
history = model_transfer.fit(train_features, y_train, batch_size=32, epochs=20,
          validation_data=(val_features, y_val), callbacks=[checkpointer],
          verbose=1, shuffle=True)

model_transfer.save('model/tr_model.h5') 

preds = le.inverse_transform(np.argmax(model_transfer.predict(test_features), axis=1))
print("\nAccuracy on Test Data: ", accuracy_score(test_y, preds))
#print("\nNumber of correctly identified imgaes: ",accuracy_score(test_y, preds, normalize=False),"\n")
#confusion_matrix(test_y, preds, labels=range(0,num_classes))

prob, pred = predict('Food-11/test/kinley1.jpg',model)
if np.max(prob)>0.8:
    pass
else:
    pred = ['unknown']     
print (prob, pred)








