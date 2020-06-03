"""Works IMGR_TF"""
from PIL import Image, ImageFont, ImageDraw  
import imageio
import imgaug as ia
import imgaug.augmenters as iaa
import glob, os
import random
import pandas as pd

def augmentation(image,degree):   
    seq = iaa.Sequential([
        iaa.Multiply((1.5, 1)),                                           # change brightness (no effect BBs). 1.0 is normal
        iaa.Affine(rotate=(degree, 0),shear=(-8, 8)),                     # scale=(0.5, 0.7), translate_px={"x": 40, "y": 60}
        iaa.AverageBlur(k=(0, 0)),
        iaa.WithColorspace(
            to_colorspace=iaa.CSPACE_HSV,
            from_colorspace=iaa.CSPACE_RGB,
            children=iaa.WithChannels(
                    0,iaa.Add((0, random.randint(0, 40))))
            )                                    
        ])
        
    image_aug = seq(image=image)
    return image_aug

def backgd_augm(bins, items, path, ratio=0.7):
    
    background = imageio.imread((bins))
    background = augmentation(background,random.randint(-40, 40))
    background = Image.fromarray(background, mode='RGB')
    background = background.resize((bg_siz,int(bg_siz*ratio)), Image.ANTIALIAS)
    
    foreground = Image.open(items)
    foreground = foreground.resize((fg_siz,int(fg_siz*ratio)), Image.ANTIALIAS)
    
    background.paste(foreground, (int(bg_siz/4), int(bg_siz/4)), foreground)  #x and y for moving bottle
    
    img_name = os.path.basename(bins)[:-4]+"_" + os.path.basename(items)[:-4]+".jpg"
    background.save(path + img_name)
    # background.show()
    return img_name

# Initialisations
items = ["PET_bottle","yoghurt_cups"] # glass_bottles
fg_siz_list = [450, 250]
bg_siz = 700

itm_df = list()  # master list

for i in range (len(items)):
    itm_mp = pd.read_excel("items/"+items[i]+"/"+items[i]+".xlsx")
    fg_siz = fg_siz_list[i]                                            # choose the require foreground size

    for bins in glob.glob("bins/*.jpg"):
        for item in glob.glob("items/"+items[i]+"/"+"*.png"):        # check if you can change it  
            
            bins=bins.replace(os.sep, '/')
            item=item.replace(os.sep, '/')
            
            print(os.path.basename(item))                           # For each item here...
    
            img_nam = [backgd_augm(bins, item, path="output/"+items[i]+"/", ratio=0.7)] # get joined name
            
            img_cod = img_nam + itm_mp[itm_mp["name"]==os.path.basename(item)[:-4]+".jpg"].values.tolist()[0] # convert to list...
            
            itm_df.append(img_cod)                                   # ...add to master list
                
pd.DataFrame.from_records(itm_df).to_excel("output/names_output.xlsx")  
        
        
        
        
        
    
