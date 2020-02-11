"""https://imgaug.readthedocs.io/en/latest/source/examples_bounding_boxes.html"""
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import imageio
import glob
import os
import pandas as pd
import numpy as np
import re

def augmentation(image,imgBBS,rotation):   
    bbs = BoundingBoxesOnImage([
        BoundingBox(x1=imgBBS[0], y1=imgBBS[1], x2=imgBBS[2], y2=imgBBS[3]),
        ], shape=image.shape)
    
    seq = iaa.Sequential([
        iaa.Multiply((1, 1)),                                           # change brightness (no effect BBs). 1.0 is normal
        iaa.Affine(rotate=(rotation, 0))                                # scale=(0.5, 0.7), translate_px={"x": 40, "y": 60}
    ])
    
    # Augment BBs and images.
    image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)
    return image_aug, bbs_aug     

# parameters
rotation = 180

# open excel
data = pd.read_excel('root_coord/VGG_via_export.xlsx', index_col=0).reset_index()
ia.seed(1)

images, file_name =[],[]
for file in glob.glob("images/originals/*.jpg"):
    images.append(imageio.imread(file))
    file_name.append(file[re.search(r'(\\)[^\\]*$',file).start()+1:])               # both file_name and images have same index and placement

BBS_cod = list()
for i in range(len(images)):  #len(file_name):
    BBS_cod.append(data[data['name'] == file_name[i]]) 
BBS_cod = pd.concat(BBS_cod)                                                        # convert to df       
    
new_bbs = list()
new_file_names = list()
for i in range(len(images)):
    image_aug, bbs_aug = augmentation(images[i],BBS_cod.iloc[i:i+1,1:5].values.tolist()[0],rotation)      
    new_name = str(file_name[i])[:-4]+'_rot_'+str(rotation)+'.jpg'
    imageio.imwrite('images/'+new_name,image_aug)
    new_bbs.append(np.array([bbs_aug.remove_out_of_image().clip_out_of_image().bounding_boxes[0].x1_int,            # clip and remove partial BBS
                             bbs_aug.remove_out_of_image().clip_out_of_image().bounding_boxes[0].y1_int,
                             bbs_aug.remove_out_of_image().clip_out_of_image().bounding_boxes[0].x2_int,
                             bbs_aug.remove_out_of_image().clip_out_of_image().bounding_boxes[0].y2_int]))          # only one BBs allowed, chosen through [0]]
    new_file_names.append(new_name.replace(os.sep, '/'))    

proc_data = pd.concat([pd.DataFrame(new_file_names),pd.DataFrame(new_bbs) ], axis=1)     
proc_data.to_excel('output.xlsx')                                # new BBS output
# can't go negative

