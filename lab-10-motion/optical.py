import cv2
import numpy as np

sourceDir = "./resource/台球.mp4"

cap = cv2.VideoCapture(sourceDir)

# 取出视频的第一帧
ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)  # 为绘制创建掩码图片
hsv[..., 1] = 255

while True:
    ret, frame = cap.read()
    if frame is None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #转为灰度图
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5,
                                        1.2, 0)  # 计算光流以获取点的新位置

    # 色调H：用角度度量，取值范围为0°～360°，从红色开始按逆时针方向计算，红色为0°，绿色为120°,蓝色为240°
    # 饱和度S：取值范围为0.0～1.0
    # 亮度V：取值范围为0.0(黑色)～1.0(白色)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2  #色调范围：0°~360°
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    cv2.imshow('frame', rgb)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('opticalfb.png', frame)
        cv2.imwrite('opticalhsv.png', rgb)
        prvs = next

cap.release()
cv2.destroyAllWindows()