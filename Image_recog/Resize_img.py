import cv2
import numpy as np
import pandas as pd
import glob

train_images = [cv2.resize(cv2.imread(file,cv2.IMREAD_GRAYSCALE),(256,192)) for file in glob.glob("train/*.png")]
test_images =  [cv2.resize(cv2.imread(file,cv2.IMREAD_GRAYSCALE),(256,192)) for file in glob.glob("test/*.png")]

for i in range(0,520):  # len(test_images)
    cv2.imwrite("../images/train_small/"+str(i)+".png", train_images[i])


for i in range(0,320):  # len(test_images)
    cv2.imwrite("../images/test_small/"+str(i)+".png", test_images[i])    
