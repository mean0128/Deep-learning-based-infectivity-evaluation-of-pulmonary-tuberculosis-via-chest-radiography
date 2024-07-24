import tensorflow as tf
from natsort import natsorted
import cv2, os
from glob import glob
import matplotlib.pyplot as plt
import threading, mritopng, cv2
import pydicom as pdc
import numpy as np

def is_image_inverted(image, threshold=63):
    height, width = image.shape[:2]
    edge_width = int(width * 0.01)
    edge_height = int(height * 0.01)
    
    edges = np.concatenate([
        image[:edge_height, :edge_width],               # 상단 왼쪽 모서리
        image[:edge_height, width-edge_width:],         # 상단 오른쪽 모서리
        image[height-edge_height:, :edge_width],        # 하단 왼쪽 모서리
        image[height-edge_height:, width-edge_width:]   # 하단 오른쪽 모서리
    ])

    mean_intensity = np.mean(edges)
    return mean_intensity > threshold
with tf.device("/GPU:0"):
    model = tf.keras.models.load_model("D:/wyj/lung_seg_valiou_088.h5")
    model.compile()

#%%
inPath = "E:/wyj/yongin_extracted/neg_png/"
outPath = "E:/wyj/yongin_extracted/neg_lung/"
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
cnt = 0
def work_func(alloc0, alloc1):
    global cnt
    for f in files[alloc0:alloc1]:
        image = cv2.imread(inPath+f)[:,:,:3]
        if is_image_inverted(image):
            image = 255-image
        image2 = cv2.resize(image, (256,256))
        image2 = np.expand_dims(image2, 0)
        with tf.device("/GPU:1"):
            res = model.predict(image2, verbose = 0)
            res = np.squeeze(res, 0)
        threshold = 0.5  # 마스크에서 임계값을 설정하세요.
        mask = (res > threshold).astype(np.uint8)
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
        print(cnt)

        
if __name__ == '__main__':
    threads = []
    for job in jobAlloc:
        t = threading.Thread(target=work_func, args = (job))
        t.start()
        threads.append(t)
        
    for thread in threads:
        thread.join()
del threads        

#%%
numStr = '37580'
image = cv2.imread("D:/wyj/sinchon_extracted/pos_png/"+numStr+".png")[:,:,:3]
if is_image_inverted(image):
    image = 255-image
plt.imshow(image)
plt.show()
image2 = cv2.resize(image, (256,256))
image2 = np.expand_dims(image2, 0)
with tf.device("/GPU:0"):
    res = model.predict(image2)
    res = np.squeeze(res, 0)
threshold = 0.5  # 마스크에서 임계값을 설정하세요.
mask = (res > threshold).astype(np.uint8)
mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
min_x = min_y = float('inf')
max_x = max_y = 0
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    min_x = min(min_x, x)
    min_y = min(min_y, y)
    max_x = max(max_x, x + w)
    max_y = max(max_y, y + h)
plt.imshow(image)
plt.show()

image[np.where(mask == 1)] += np.array((32,32,0), dtype = np.uint8)
cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (255,0,0), thickness = 10)
cv2.imwrite("D:/wyj/"+numStr+"_seg_crop.png", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.imshow(image)
plt.show()
    
plt.imshow(image[min_y:max_y, min_x:max_x])
plt.show()
cv2.imwrite("D:/wyj/"+numStr+"_res.png", cv2.cvtColor(image[min_y:max_y, min_x:max_x], cv2.COLOR_BGR2RGB))
