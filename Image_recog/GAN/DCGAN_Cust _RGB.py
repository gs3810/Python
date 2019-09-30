import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    
    train_images = [((cv2.resize(cv2.imread(file),(128,128))).astype(np.float32) - 127.5)/127.5 for file in glob.glob(train_path)]
    test_images = [cv2.resize(cv2.imread(file),(128,128)) for file in glob.glob(test_path)]
    
    x_train = np.stack(train_images,axis=0)
    x_test = np.stack(test_images,axis=0)
    
    y_train = np.ones(len(train_images))
    y_test = np.ones(len(test_images))

    return x_train, y_train, x_test, y_test

def adam_optimizer():
    return adam(lr=0.0002, beta_1=0.5)

def create_dcgenerator():
    img_size = [img_wid, img_hg]
    upsample_layers = 4
    starting_filters = 64
    kernel_size = 3                                  
    noise_shape = (25,)

    model = Sequential()
    
    model.add(
    Dense(starting_filters*(img_size[0] // (2**upsample_layers))*(img_size[1] // (2**upsample_layers)),
          activation="relu", input_shape=noise_shape))
    model.add(Reshape(((img_size[0] // (2**upsample_layers)),
                       (img_size[1] // (2**upsample_layers)),
                       starting_filters)))
    model.add(BatchNormalization(momentum=0.8))

    model.add(UpSampling2D())  # 8x8 -> 16x16
    model.add(Conv2D(512, kernel_size=6, padding="same"))   # large kernel  # change the numbers of layers accordingly
    model.add(Activation("relu"))
    model.add(BatchNormalization(momentum=0.8))

    model.add(UpSampling2D())  # 16x16 -> 32x32
    model.add(Conv2D(256, kernel_size=5, padding="same"))   # mid kernel
    model.add(Activation("relu"))
    model.add(BatchNormalization(momentum=0.8))
    
    model.add(UpSampling2D())  # 32x32 -> 64x64
    model.add(Conv2D(128, kernel_size=4, padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(momentum=0.8))
    
    model.add(UpSampling2D())  # 64x64 -> 128x128
    model.add(Conv2D(64, kernel_size=kernel_size, padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(momentum=0.8))
    
    model.add(Conv2D(64, kernel_size=kernel_size, padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(momentum=0.8))
    
    model.add(Conv2D(channels, kernel_size=kernel_size, padding="same"))
    model.add(Activation("tanh"))

    model.summary()
    model.compile(loss='binary_crossentropy', optimizer=adam_optimizer())
    return model

def create_discriminator():
    img_shape = (img_wid, img_hg, channels)
    kernel_size = 3  
    
    discriminator=Sequential()
    
    discriminator.add(Conv2D(32, kernel_size=kernel_size, strides=1, input_shape=img_shape, padding="same"))    # 64x64 <- 128x128
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Dropout(0.25))
    
    discriminator.add(Conv2D(64, kernel_size=kernel_size, strides=1, padding="same"))  # 32x32 <- 64x64
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Dropout(0.25))
     
    discriminator.add(Conv2D(64, kernel_size=kernel_size, strides=1, padding="same"))  # 16x16 <- 32x32
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Dropout(0.25))
    
    discriminator.add(Conv2D(64, kernel_size=kernel_size, strides=1, padding="same"))  # 8x8 <- 16x16
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Dropout(0.25))
    
    discriminator.add(Dense(units=256))
    discriminator.add(LeakyReLU(0.2))
    
    discriminator.add(Flatten())
    discriminator.add(Dense(1, activation='sigmoid'))
    
    discriminator.compile(loss='binary_crossentropy', optimizer=adam_optimizer(), metrics=["accuracy"])
    return discriminator

def create_gan(discriminator, generator):
    discriminator.trainable=False
    gan_input = Input(shape=(25,))
    x = generator(gan_input)
    gan_output= discriminator(x)
    gan= Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer='adam')
    return gan

def plot_generated_images(epoch, generator, examples=25, dim=(5,5), figsize=(10,10)):
    noise= np.random.normal(loc=0, scale=1, size=[examples, 25])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(25,img_wid,img_hg,channels)
    
    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        generated_images[i] = (generated_images[i]*127.5)
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(cv2.cvtColor(generated_images[i], cv2.COLOR_BGR2RGB))
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('gan_generated_image %d.png' %epoch)
    
def training(epochs=1, batch_size=128):
    
    #Loading the data
    (X_train, y_train, X_test, y_test) = load_custom_images()
    batch_count = X_train.shape[0] / batch_size
    
    # Creating GAN
    generator= create_dcgenerator()
    discriminator= create_discriminator()
    gan = create_gan(discriminator, generator)

    for e in range(1,epochs+1 ):
        print("Epoch %d" %e)
        for _ in tqdm(range(batch_size)):
        #generate  random noise as an input  to  initialize the  generator
            noise= np.random.normal(0,1, [batch_size, 25])
            
            # Generate fake MNIST images from noised input
            generated_images = generator.predict(noise).reshape(batch_size,img_wid*img_hg,channels)
            
            # Get a random set of  real images
            image_batch =X_train[np.random.randint(low=0,high=X_train.shape[0],size=batch_size)].reshape(batch_size,img_wid*img_hg,channels)

            #Construct different batches of  real and fake data, with INVERSION for generalization            
            if np.random.randint(10)<8:
                X= np.concatenate([image_batch, generated_images])
                print("normal")    
            else:
                X= np.concatenate([image_batch, generated_images])
                print("inverted")                
            
            X = X.reshape(batch_size*2,img_wid,img_hg,channels)
            
            # Noisy labels for generated and real data
            y_dis = np.random.randint(0,40, size=2*batch_size)/100
            y_dis[:batch_size]=np.random.randint(60,100, size=batch_size)/100
            
            #Pre train discriminator on  fake and real data  before starting the gan. 
            discriminator.trainable=True
            discriminator.train_on_batch(X, y_dis)
            
            #Tricking the noised input of the Generator as real data
            noise= np.random.normal(0,1, [batch_size, 25])
            y_gen = np.ones(batch_size)
            
            # During the training of gan, the weights of discriminator should be fixed. 
            discriminator.trainable=False
            
            #training  the GAN by alternating the training of the Discriminator and training the chained GAN model with Discriminator’s weights freezed.
            gan.train_on_batch(noise, y_gen)
            
        if e == 1 or e % 5 == 0:           
            plot_generated_images(e, generator)  
            
global img_wid, img_hg, train_path, test_path, channels

train_path = "images/train/*.jpg"
test_path = "images/test/*.jpg"

img_wid, img_hg = 128,128
channels = 3  

(X_train, y_train,X_test, y_test) = load_custom_images()

g = create_dcgenerator()
#g.summary()

d = create_discriminator()
#d.summary()

gan = create_gan(d,g)
gan.summary()

training(40,10)


"""
train_images = [((cv2.resize(cv2.imread(file),(128,128))).astype(np.float32))/127.5 for file in glob.glob(train_path)]
test_images = [cv2.resize(cv2.imread(file),(128,128)) for file in glob.glob(test_path)]


check = train_images[1]
plt.imshow(cv2.cvtColor(check, cv2.COLOR_BGR2RGB))

x_train = np.stack(train_images,axis=0)
#x_train = np.rollaxis(np.dstack(train_images),-1)
#x_train = x_train.reshape(len(train_images),img_wid*img_hg)  # bring to 2D
x_test = np.stack(test_images,axis=0)
 
"""


