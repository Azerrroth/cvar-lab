import cv2
import os
import numpy as np

filename = "finger.jpeg"
img_size = (640, 640)


def autoDetectPoints(img):

    # 转为灰度单通道 [[255 255],[255 255]]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 二值化图像
    ret, img_b = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    # 图像出来内核大小，相当于PS的画笔粗细
    kernel = np.ones((5, 5), np.uint8)
    # 图像膨胀
    img_dilate = cv2.dilate(img_b, kernel, iterations=8)
    # 图像腐蚀
    img_erode = cv2.erode(img_dilate, kernel, iterations=3)
    # 寻找轮廓
    contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    # 绘制轮廓
    # cv2.drawContours(img, contours, -1, (255, 0, 255), 1)

    # 一般会找到多个轮廓，这里因为我们处理成只有一个大轮廓
    contour = contours
    docCnt = None
    if len(contour) > 0:
        contour = sorted(contour, key=cv2.contourArea,
                         reverse=True)  # 根据轮廓面积从大到小排序
        for c in contour:
            peri = cv2.arcLength(c, True)  # 计算轮廓周长
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)  # 轮廓多边形拟合
            # 轮廓为4个点表示找到纸张
            if len(approx) == 4:
                docCnt = approx
                break

    # 每个轮廓进行多边形拟合
    # approx = cv2.approxPolyDP(contour, 150, True)
    # 绘制拟合结果，这里返回的点的顺序是：左上，左下，右下，右上
    # cv2.polylines(img, [docCnt], True, (0, 255, 0), 2)

    # 寻找最小面积矩形
    # rect = cv2.minAreaRect(contour[0])
    # 转化为四个点，这里四个点顺序是：左上，右上，右下，左下
    # box = np.int0(cv2.boxPoints(rect))
    # 绘制矩形结果
    # cv2.drawContours(img, [box], 0, (0, 66, 255), 2)

    return docCnt


if __name__ == '__main__':
    img = cv2.imread(os.path.join("resource", filename))
    approx = autoDetectPoints(img)
    # 同一成一个顺序：左上，左下，右下，右上
    src = np.float32(approx)
    dst = np.float32([[0, 0], [0, img_size[1]], [img_size[0], img_size[1]],
                      [img_size[0], 0]])
    print(src, dst)
    M = cv2.getPerspectiveTransform(src, dst)
    pers = cv2.warpPerspective(img, M, img_size, borderValue=(255, 255, 255))
    # fixed = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
    cv2.imshow('Fixed finger print', pers)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
