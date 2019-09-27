import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.layers import Dense, Dropout, Input, UpSampling2D, Conv2D, Activation, BatchNormalization, Reshape, LeakyReLU, Flatten, ZeroPadding2D
from keras.models import Model,Sequential
from keras.datasets import mnist
from tqdm import tqdm
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import adam
import cv2
import glob

def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()    
    x_train = (x_train.astype(np.float32) - 127.5)/127.5

    x_train = x_train.reshape(60000, 784)
    return (x_train, y_train, x_test, y_test)

def load_custom_images():
    
    train_images = [((cv2.resize(cv2.imread(file,cv2.IMREAD_GRAYSCALE),(64,64))).astype(np.float32) - 127.5)/127.5 for file in glob.glob("images/train/*.jpg")]
    test_images = [cv2.resize(cv2.imread(file,cv2.IMREAD_GRAYSCALE),(64,64)) for file in glob.glob("images/test/*.jpg")]
    
    x_train = np.dstack(train_images).reshape(len(train_images),img_wid*img_hg )
    x_test = np.dstack(test_images).reshape(len(test_images),img_wid,img_hg )
    
    y_train = np.ones(len(train_images))
    y_test = np.ones(len(test_images))

    return x_train, y_train, x_test, y_test

def adam_optimizer():
    return adam(lr=0.0002, beta_1=0.5)

def create_generator():
    generator=Sequential()
    
    generator.add(Dense(units=128,input_dim=100))
    generator.add(LeakyReLU(0.2))
    
    generator.add(Dense(units=256))
    generator.add(LeakyReLU(0.2))
    
    generator.add(Dense(units=512))
    generator.add(LeakyReLU(0.2))
    
    generator.add(Dense(units=1024))
    generator.add(LeakyReLU(0.2))
    
    generator.add(Dense(units=img_wid*img_hg, activation='tanh'))
    
    generator.compile(loss='binary_crossentropy', optimizer=adam_optimizer())
    return generator

def create_dcgenerator():
    img_size = [img_wid, img_hg]
    upsample_layers = 2
    starting_filters = 14
    kernel_size = 3                                      # No RGB channels
    noise_shape = (100,)

    model = Sequential()
    
    model.add(
    Dense(starting_filters*(img_size[0] // (2**upsample_layers))*(img_size[1] // (2**upsample_layers)),
          activation="relu", input_shape=noise_shape))
    model.add(Reshape(((img_size[0] // (2**upsample_layers)),
                       (img_size[1] // (2**upsample_layers)),
                       starting_filters)))
    model.add(BatchNormalization(momentum=0.8))

    model.add(UpSampling2D())  # 7x7 -> 14x14
    model.add(Conv2D(128, kernel_size=kernel_size, padding="same"))     # change the numbers of layers accordingly
    model.add(Activation("relu"))
    model.add(BatchNormalization(momentum=0.8))

    model.add(UpSampling2D())  # 14x14 -> 28x28
    model.add(Conv2D(64, kernel_size=kernel_size, padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(momentum=0.8))

    model.add(Conv2D(32, kernel_size=kernel_size, padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(momentum=0.8))
    
    model.add(Conv2D(1, kernel_size=kernel_size, padding="same"))
    model.add(Activation("tanh"))

    model.summary()
    model.compile(loss='binary_crossentropy', optimizer=adam_optimizer())
    return model

def create_discriminator_old():
    discriminator=Sequential()
    discriminator.add(Dense(units=1024,input_dim=img_wid*img_hg))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
       
    discriminator.add(Dense(units=512))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
    
    discriminator.add(Dense(units=256))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
       
    discriminator.add(Dense(units=128))
    discriminator.add(LeakyReLU(0.2))
    
    discriminator.add(Dense(units=1, activation='sigmoid'))
    
    discriminator.compile(loss='binary_crossentropy', optimizer=adam_optimizer())
    return discriminator

def create_discriminator():
    img_shape = (img_wid, img_hg, 1)
    kernel_size = 3  
    
    discriminator=Sequential()
    
    discriminator.add(Conv2D(32, kernel_size=kernel_size, strides=1, input_shape=img_shape, padding="same"))    # 28x28 -> 14x14
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Dropout(0.25))
    
    discriminator.add(Conv2D(64, kernel_size=kernel_size, strides=1, padding="same"))  # 14x14 -> 7x7
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Dropout(0.25))
       
    discriminator.add(Dense(units=256))
    discriminator.add(LeakyReLU(0.2))
    
    discriminator.add(Flatten())
    discriminator.add(Dense(1, activation='sigmoid'))
    
    discriminator.compile(loss='binary_crossentropy', optimizer=adam_optimizer())
    return discriminator

def create_gan(discriminator, generator):
    discriminator.trainable=False
    gan_input = Input(shape=(100,))
    x = generator(gan_input)
    gan_output= discriminator(x)
    gan= Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer='adam')
    return gan

def plot_generated_images(epoch, generator, examples=100, dim=(10,10), figsize=(10,10)):
    noise= np.random.normal(loc=0, scale=1, size=[examples, 100])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(100,img_wid,img_hg)
    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('gan_generated_image %d.png' %epoch)
    
def training(epochs=1, batch_size=128):
    
    #Loading the data
    (X_train, y_train, X_test, y_test) = load_data()
    batch_count = X_train.shape[0] / batch_size
    
    # Creating GAN
    generator= create_dcgenerator()
    discriminator= create_discriminator()
    gan = create_gan(discriminator, generator)

    for e in range(1,epochs+1 ):
        print("Epoch %d" %e)
        for _ in tqdm(range(batch_size)):
        #generate  random noise as an input  to  initialize the  generator
            noise= np.random.normal(0,1, [batch_size, 100])
            
            # Generate fake MNIST images from noised input
            generated_images = generator.predict(noise).reshape(batch_size,img_wid*img_hg,1)
            
            # Get a random set of  real images
            image_batch =X_train[np.random.randint(low=0,high=X_train.shape[0],size=batch_size)].reshape(batch_size,img_wid*img_hg,1)

            #Construct different batches of  real and fake data 
            X= np.concatenate([image_batch, generated_images])
            X = X.reshape(batch_size*2,img_wid,img_hg,1)
            
            # Labels for generated and real data
            y_dis=np.zeros(2*batch_size)
            y_dis[:batch_size]=0.9
            
            #Pre train discriminator on  fake and real data  before starting the gan. 
            discriminator.trainable=True
            discriminator.train_on_batch(X, y_dis)
            
            #Tricking the noised input of the Generator as real data
            noise= np.random.normal(0,1, [batch_size, 100])
            y_gen = np.ones(batch_size)
            
            # During the training of gan, 
            # the weights of discriminator should be fixed. 
            # We can enforce that by setting the trainable flag
            discriminator.trainable=False
            
            #training  the GAN by alternating the training of the Discriminator 
            #and training the chained GAN model with Discriminator’s weights freezed.
            gan.train_on_batch(noise, y_gen)
            
        if e == 1 or e % 20 == 0:           
            plot_generated_images(e, generator)  
            
global img_wid, img_hg             
img_wid, img_hg = 28,28  

(X_train, y_train,X_test, y_test) = load_data()
#(X_train, y_train,X_test, y_test) = load_custom_images()

#g = create_generator()
g = create_dcgenerator()
#g.summary()

d = create_discriminator()
#d.summary()

gan = create_gan(d,g)
gan.summary()

training(400,128)

"""
discriminator1= create_discriminator_old()
discriminator2= create_discriminator()

generator1= create_dcgenerator()
generator2= create_generator()

noise= np.random.normal(0,1, [128, 100])
#generated_images = generator1.predict(noise)

generated_images = generator2.predict(noise).reshape(128,784,1)


image_batch =X_train[np.random.randint(low=0,high=X_train.shape[0],size=128)].reshape(128,784,1)
#Construct different batches of  real and fake data 
X= np.concatenate([image_batch, generated_images])

# Labels for generated and real data
y_dis=np.zeros(2*128)
y_dis[:128]=0.9


X = X.reshape(256,28,28,1)
#Pre train discriminator on  fake and real data  before starting the gan. 
discriminator2.trainable=True
discriminator2.train_on_batch(X, y_dis)

# Try DCGAn https://github.com/DataSnaek/DCGAN-Keras/blob/master/DCGAN.py 
"""

