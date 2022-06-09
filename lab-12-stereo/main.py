import cv2
import numpy as np

sourceDir = "./resource"

imgL = cv2.imread(sourceDir + "/left.png")
imgR = cv2.imread(sourceDir + "/right.png")

imgLG = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
imgRG = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

stereo = cv2.StereoBM_create(numDisparities=128, blockSize=7)
disp = stereo.compute(imgLG, imgRG).astype(np.float32) / 16.0

cv2.imshow("disparity", (disp - min(disp.min(), 0)) /
           (max(disp.max(), 0) - min(disp.min(), 0)))
while 1:
    key = cv2.waitKey(1)
    if key > 0:
        break
cv2.destroyAllWindows()
