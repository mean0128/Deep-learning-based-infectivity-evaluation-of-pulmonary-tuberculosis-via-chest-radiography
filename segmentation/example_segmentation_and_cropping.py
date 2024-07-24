import cv2, os
import numpy as np
import tensorflow as tf
from natsort import natsorted
from tensorflow_addons.optimizers import AdamW
from keras_unet_collection.activations import GELU
from segmentation_models.losses import bce_dice_loss
from keras_unet_collection.transformer_layers import patch_extract
from keras_unet_collection.transformer_layers import patch_embedding

inPath = "input folder path"
outPath = "output folder path"
threshold = 0.5 # segmentation probability threshold
model = tf.keras.models.load_model("segmentation model file path.h5", # model file path should be modified 
                                   custom_objects={'binary_crossentropy_plus_dice_loss' : bce_dice_loss,
                                                   'patch_embedding' : patch_embedding,
                                                   'patch_extract' : patch_extract,
                                                   'Addons>AdamW' : AdamW,
                                                   'GELU' : GELU,
                                                   })

def is_image_inverted(image, threshold=63):
    height, width = image.shape[:2]
    edge_width = int(width * 0.01)
    edge_height = int(height * 0.01)
    
    edges = np.concatenate([
        image[:edge_height, :edge_width],
        image[:edge_height, width-edge_width:],
        image[height-edge_height:, :edge_width],
        image[height-edge_height:, width-edge_width:]
    ])

    mean_intensity = np.mean(edges)
    return mean_intensity > threshold

#%%
files = natsorted(os.listdir(inPath))
totalN = len(os.listdir(inPath))
threads = totalN 
cnt = 0
while(threads > 32):
    threads /= 2
threads = int(threads)
allocJobs = int(totalN / (threads - 1))
jobAlloc = [[i*allocJobs, (i+1)*allocJobs] for i in range(threads -1)]
jobAlloc.append([ jobAlloc[len(jobAlloc)-1][1], totalN ])
def work_func(alloc0, alloc1):
    for f in files[alloc0:alloc1]:
        image = cv2.imread(inPath, cv2.IMREAD_GRAYSCALE)
        if is_image_inverted(image):
            image = 255-image
        image2 = cv2.resize(image, (512,512))/255.
        image2 = np.expand_dims(image2, 0)
        res = model.predict(image2, verbose = 0)
        res = np.squeeze(res, 0)
        mask = (res > threshold).astype(np.uint8)
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2] # largest 2 contours (left, right lung)
        min_x = min_y = float('inf')
        max_x = max_y = 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x + w)
            max_y = max(max_y, y + h)
        cv2.imwrite(outPath + f, image[min_y:max_y, min_x:max_x])
        cnt += 1
        
if __name__ == '__main__':
    threads = []
    for job in jobAlloc:
        t = threading.Thread(target=work_func, args = (job))
        t.start()
        threads.append(t)
        
    for thread in threads:
        thread.join()
del threads
