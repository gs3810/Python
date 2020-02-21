import pandas as pd
import numpy as np
import re

data = pd.read_csv("annot.csv")

img_name = data.iloc[:,[0]]
label = data.iloc[:,[4]]
coord = data.iloc[:,[5]]

coord_list= list()
for i in range(coord.shape[0]):
    string = coord.iloc[i,0]
    coord_list.append(re.findall(r'\d+',string))

coord_df = pd.DataFrame(coord_list)
img_coord = pd.concat([img_name,label,coord_df], axis=1)

for i in range(img_coord.shape[0]):
    file = img_coord.iloc[i,0][:-4]             # remove jpg
    np.savetxt('ground_truths/'+file+'.txt', img_coord.iloc[i:i+1,1:].astype(int), delimiter=' ',fmt='%1.0f')
    

