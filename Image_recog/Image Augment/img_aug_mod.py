"""https://imgaug.readthedocs.io/en/latest/source/examples_bounding_boxes.html"""
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import imageio
import glob
import os
import pandas as pd

def augmentation(image,imgBBS):   
    rotation = 270
    bbs = BoundingBoxesOnImage([
        BoundingBox(x1=imgBBS[0], y1=imgBBS[1], x2=imgBBS[2], y2=imgBBS[3]),
        ], shape=image.shape)
    
    seq = iaa.Sequential([
        iaa.Multiply((1, 1)),                                   # change brightness (no effect BBs). 1.0 is normal
        iaa.Affine(rotate=(rotation, 0))                        # scale=(0.5, 0.7), translate_px={"x": 40, "y": 60}
    ])
    
    # Augment BBs and images.
    image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)

    return image_aug, bbs_aug     

def print_BBS(bbs_aug):
    for i in range(len(bbs_aug.bounding_boxes)):
        after = bbs_aug.bounding_boxes[i]
    #    after = list(after)
        print("BB %d:(%.4f, %.4f, %.4f, %.4f)" % (
            i,after.x1, after.y1, after.x2, after.y2))

# open excel
data = pd.read_excel('root/VGG_via_export.xlsx', index_col=0).reset_index()
ia.seed(1)

images, file_name =[],[]
for file in glob.glob("images/*.jpg"):
    images.append(imageio.imread(file))
    file_name.append(file)                                                   # both file_name and images have same index and placement

BBS_cod = list()
for i in range(2):  #len(file_name):
    BBS_cod.append(data[data['name'] == file_name[i].replace(os.sep, '/')]) 
BBS_cod = pd.concat(BBS_cod)                                                 # convert to df       
    
new_bbs = list()
for i in range(len(images)):
    image_aug, bbs_aug = augmentation(images[i],BBS_cod.iloc[i:i+1,1:5].values.tolist()[0])      
    imageio.imwrite('images/new/new_'+str(i)+'.jpg',image_aug)
    new_bbs.append(list(bbs_aug.bounding_boxes[0]))                         # only one BBs allowed

new_bbs = pd.DataFrame(new_bbs)                                             # new BBS output
# can't go negative

