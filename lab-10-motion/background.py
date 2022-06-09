import numpy as np
import cv2

sourceDir = "./resource/台球.mp4"

cap = cv2.VideoCapture(sourceDir)

fgbg = cv2.createBackgroundSubtractorMOG2()  #背景分割

while (1):
    ret, frame = cap.read()
    if frame is None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    fgmask = fgbg.apply(frame)  #应用

    cv2.imshow('frame', fgmask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()