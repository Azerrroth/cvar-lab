# %%
import cv2
'''
要求:

(1)打开视频设备, 并显示视频
(2)打开视频或图像文件, 并显示
(3)在视频(或图像)上叠加自己的学号和姓名
'''
# %% 图片展示

img = cv2.imread('resource/cat.jpeg', cv2.IMREAD_COLOR)

# 调用cv.putText()添加文字，姓名、学号
text = "{} {}".format("Hello", "World")

cv2.putText(img, text, (0, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

# 创建窗口并展示图片
cv2.imshow('image', img)

# 等待任意一个按键按下
cv2.waitKey(0)
# 关闭所有的窗口
cv2.destroyAllWindows()

# %% 视频设备

# 创建一个视频设备对象
cap = cv2.VideoCapture(0)
# 判断视频设备是否创建成功
if not cap.isOpened():
    print('open failed!')
    exit(0)
# 循环读取视频帧
while True:
    # 读取视频帧
    ret, frame = cap.read()
    # 判断视频帧是否读取成功
    if not ret:
        print('read failed!')
        break
    cv2.putText(frame, text, (0, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255),
                2)

    # 显示视频帧
    cv2.imshow('video', frame)
    # 等待任意一个按键按下
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# 释放视频设备
cap.release()
# 关闭所有的窗口
cv2.destroyAllWindows()
