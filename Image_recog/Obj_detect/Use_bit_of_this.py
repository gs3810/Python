import sys
import argparse
from yolo import YOLO, detect_video
from PIL import Image
import tensorflow as tf
import pandas as pd
import numpy as np
import re
print(tf.__version__)

"""mAP https://github.com/rafaelpadilla/Object-Detection-Metrics"""

def detect_img(yolo,inp_imgs):
 
    out_boxes, out_scores, out_classes = list(),list(),list()
    for i in range(0,len(inp_imgs)):                        # while True:
        img = inp_imgs[i]                                   #input('Input image filename:')  # file path e.g. images/keells_6.jpg (no 'commas')
        try:
            image = Image.open(img)                         # try RESIZING + img AUGMENTATION
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image, boxes, scores, classes = yolo.detect_image(image)
            out_boxes.append(boxes)
            out_scores.append(scores)
            out_classes.append(classes)
            r_image.show()
            r_image.save("detection_images/detect_"+str(i)+".jpg")
            
    return out_boxes, out_scores, out_classes 
#    yolo.close_session()"

FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    
    parser.add_argument(                    # turned default = True
        '--image', default=True, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )

    FLAGS = parser.parse_args()
    if True:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)

        # detect images        
        inp_imgs = ['images/test_images/test_2.jpg','images/test_images/test_3.jpg',
                    'images/test_images/test_4.jpg','images/test_images/test_6.jpg',
                    'images/test_images/test_7.jpg']
        
        score = detect_img(YOLO(**vars(FLAGS)),inp_imgs)

        # convert everything into df        
        bbs = np.array(score[0]).reshape(-1,4)              # reshape to ...x4
        # convert to x,y,w,h and reorganize format                                
        bbs[:,2:3] = bbs[:,2:3] - bbs[:,0:1] 
        bbs[:,3:4] = bbs[:,3:4] - bbs[:,1:2]
        bbs = bbs[:,[1,0,3,2]]
        
        df_bbs = pd.DataFrame(bbs)        
        df_conf = pd.DataFrame(score[1])
        df_class = pd.DataFrame(score[2])     
        df_score = pd.concat([df_class,df_conf,df_bbs], axis=1).round(0)
        
        for i in range(len(inp_imgs)):
            file = inp_imgs[i]
            file = file[re.search(r'(/)[^/]*$',file).start()+1:][:-4]                   # remove '.jpg'
            np.savetxt('logs/score_labels/'+file+'.txt', df_score.iloc[i:i+1,:], delimiter=' ',fmt='%1.2f')
#        df_score.to_excel("logs/score_labels/Detected_labels.xlsx")
            
            

