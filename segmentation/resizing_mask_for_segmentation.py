import cv2, os
import threading
targetPath = "path to folder where the mask is"
outPath = "path to the folder where the mask will be saved "
files = os.listdir(targetPath)
totalN = len(os.listdir(targetPath))
threads = totalN 
while(threads > 64):
    threads /= 2
threads = int(threads)
allocJobs = int(totalN / (threads - 1))
jobAlloc = [[i*allocJobs, (i+1)*allocJobs] for i in range(threads -1)]
jobAlloc.append([ jobAlloc[len(jobAlloc)-1][1], totalN ])
cnt = 0

def work_func(alloc0, alloc1):
    global cnt
    for f in os.listdir(targetPath)[alloc0:alloc1]:
        img = cv2.imread(targetPath + f, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (512,512))
        img[img < 64] = 0 # original segmentation mask is composed of 0 and 255
        img[img>=64] = 1
        cv2.imwrite(outPath + f, img)
        print(cnt)
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
