"""https://imgaug.readthedocs.io/en/latest/source/examples_bounding_boxes.html"""
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import imageio
import glob
import os
import pandas as pd

# open excel
data = pd.read_excel('root/VGG_via_export.xlsx', index_col=0).reset_index()
ia.seed(1)

images, file_name =[],[]
for file in glob.glob("images/*.jpg"):
    images.append(imageio.imread(file))
    file_name.append(file)


BBS_cod = pd.DataFrame()
for i in range(1):  #len(file_name):
    BBS_cod.append(data[data['name'] == file_name[i].replace(os.sep, '/')]) # still not working

# build for _loop and add

#---------------------------------------

image = imageio.imread('images/coca_env_3.jpg')
imgBBS = [137,44,282,441]

bbs = BoundingBoxesOnImage([
    BoundingBox(x1=imgBBS[0], y1=imgBBS[1], x2=imgBBS[2], y2=imgBBS[3]),
#    BoundingBox(x1=150, y1=80, x2=200, y2=130)
], shape=image.shape)

seq = iaa.Sequential([
    iaa.Multiply((1, 1)),     # change brightness (no effect BBs). 1.0 is normal
    iaa.Affine(rotate=(180, 0)) #scale=(0.5, 0.7), translate_px={"x": 40, "y": 60}
])

# Augment BBs and images.
image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)

# print coordinates before/after augmentation (see below). Use .x1_int, .y_int, ... to get integer coordinates
for i in range(len(bbs.bounding_boxes)):
    before = bbs.bounding_boxes[i]
    after = bbs_aug.bounding_boxes[i]
#    after = list(after)
    print("BB %d: (%.4f, %.4f, %.4f, %.4f) -> (%.4f, %.4f, %.4f, %.4f)" % (
        i,
        before.x1, before.y1, before.x2, before.y2,
        after.x1, after.y1, after.x2, after.y2)
    )

imageio.imwrite('images/new.jpg',image_aug)

## image with BBs before/after augmentation (shown below)
#image_before = bbs.draw_on_image(image, size=2)
#image_after = bbs_aug.draw_on_image(image_aug, size=2, color=[0, 0, 255])
#



