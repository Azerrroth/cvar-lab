import cv2
import numpy as np
import os


def waterShed(sourceDir):
    # 读取图片
    img = cv2.imread(sourceDir)
    # 原图灰度处理,输出单通道图片
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 二值化处理Otsu算法
    reval_O, dst_Otsu = cv2.threshold(gray, 0, 255,
                                      cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # 二值化处理Triangle算法
    reval_T, dst_Tri = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_TRIANGLE)
    # 滑动窗口尺寸
    kernel = np.ones((3, 3), np.uint8)
    # 形态学处理:开处理,膨胀边缘
    opening = cv2.morphologyEx(dst_Tri, cv2.MORPH_OPEN, kernel, iterations=2)
    # 膨胀处理背景区域
    dilate_bg = cv2.dilate(opening, kernel, iterations=3)
    # 计算开处理图像到邻域非零像素距离
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    # 正则处理
    norm = cv2.normalize(dist_transform, 0, 255, cv2.NORM_MINMAX)
    # 阈值处理距离图像,获取图像前景图
    retval_D, dst_fg = cv2.threshold(dist_transform,
                                     0.5 * dist_transform.max(), 255, 0)
    # 前景图格式转换
    dst_fg = np.uint8(dst_fg)
    # 未知区域计算:背景减去前景
    unknown = cv2.subtract(dilate_bg, dst_fg)
    cv2.imshow("Difference value", unknown)
    cv2.imwrite('./images/saved/unknown_reginon.png', unknown)
    # 处理连接区域
    retval_C, marks = cv2.connectedComponents(dst_fg)
    cv2.imshow('Connect marks', marks)
    cv2.imwrite('./images/saved/connect_marks.png', marks)
    # 处理掩模
    marks = marks + 1
    marks[unknown == 255] = 0
    cv2.imshow("marks undown", marks)
    # 分水岭算法分割
    marks = cv2.watershed(img, marks)
    # 绘制分割线
    img[marks == -1] = [255, 0, 255]
    cv2.imshow("Watershed", img)
    cv2.imwrite('./images/saved/watershed.png', img)
    cv2.waitKey(0)


def cutImage(sourceDir):
    # 读取图片
    img = cv2.imread(sourceDir)
    # 灰度化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 高斯模糊处理:去噪(效果最好)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    # Sobel计算XY方向梯度
    gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0)
    gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1)
    # 计算梯度差
    gradient = cv2.subtract(gradX, gradY)
    # 绝对值
    gradient = cv2.convertScaleAbs(gradient)
    # 高斯模糊处理:去噪(效果最好)
    blured = cv2.GaussianBlur(gradient, (9, 9), 0)
    # 二值化
    _, dst = cv2.threshold(blured, 90, 255, cv2.THRESH_BINARY)
    # 滑动窗口
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (107, 76))
    # 形态学处理:形态闭处理(腐蚀)
    closed = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, kernel)
    # 腐蚀与膨胀迭代
    closed = cv2.erode(closed, None, iterations=4)
    closed = cv2.dilate(closed, None, iterations=4)
    # 获取轮廓
    cnts, _ = cv2.findContours(closed.copy(), cv2.RETR_LIST,
                               cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    rect = cv2.minAreaRect(c)
    box = np.int0(cv2.boxPoints(rect))
    draw_img = cv2.drawContours(img.copy(), [box], -1, (0, 0, 255), 3)
    cv2.imshow("Box", draw_img)
    cv2.imwrite('./images/saved/monkey.png', draw_img)
    cv2.waitKey(0)


def grab_cut(sourceDir):
    # 读取图片
    img = cv2.imread(sourceDir)
    # 分割的矩形区域
    rect = (96, 1, 359, 358)
    # 背景模式,必须为1行,13x5列
    bgModel = np.zeros((1, 65), np.float64)
    # 前景模式,必须为1行,13x5列
    fgModel = np.zeros((1, 65), np.float64)
    # 图像掩模,取值有0,1,2,3
    mask = np.zeros(img.shape[:2], np.uint8)
    # grabCut处理,GC_INIT_WITH_RECT模式
    cv2.grabCut(img, mask, rect, bgModel, fgModel, 4, cv2.GC_INIT_WITH_RECT)
    # grabCut处理,GC_INIT_WITH_MASK模式
    # cv2.grabCut(img, mask, rect, bgModel, fgModel, 4, cv2.GC_INIT_WITH_MASK)
    # 将背景0,2设成0,其余设成1
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    # 重新计算图像着色,对应元素相乘
    img = img * mask2[:, :, np.newaxis]
    cv2.imwrite(os.path.join(os.path.dirname(__file__), "segment.png"), img)
    cv2.imshow("Result", img)
    cv2.waitKey(0)


if __name__ == "__main__":
    sourceDir = "./resource/segment.jpeg"
    # waterShed(sourceDir)
    grab_cut(sourceDir)
    # cutImage(sourceDir)