import cv2
import numpy as np
import random

# 
def average_white_balance(img):
    '''
    
    '''
    # 分离三个通道
    b, g, r = cv2.split(img)
    # 计算每个通道的平均值
    avg_b = np.mean(b)
    avg_g = np.mean(g)
    avg_r = np.mean(r)
    # 计算白平衡系数
    avg_w = (avg_b + avg_g + avg_r) / 3
    # 白平衡
    # b = (b * (avg_w / avg_b)).astype(np.uint8)
    # g = (g * (avg_w / avg_g)).astype(np.uint8)
    # r = (r * (avg_w / avg_r)).astype(np.uint8)

    # b, g, r = cv2.split(img)
    r = cv2.addWeighted(src1=r, alpha=avg_w / avg_r, src2=0, beta=0, gamma=0)
    g = cv2.addWeighted(src1=g, alpha=avg_w / avg_g, src2=0, beta=0, gamma=0)
    b = cv2.addWeighted(src1=b, alpha=avg_w / avg_b, src2=0, beta=0, gamma=0)
    # 这里使用 cv2.addWeighted 而不用 numpy 直接进行加减，主要是因为 cv2加法是饱和操作，而 numpy 加法是溢出操作
    # 例：
    # np: 250 + 10 = 260 > 255 = 4
    # cv: 250 + 10 = 260 > 255 = 255
    
    img = cv2.merge([b, g, r])
    return img

def perfect_reflection(img):
    '''
    完美反射
    STEP 1：计算每个像素的R\G\B之和
    STEP 2：按R+G+B值的大小计算出其前Ratio%的值作为参考点的的阈值T
    STEP 3：对图像中的每个点，计算其中R+G+B值大于T的所有点的R\G\B分量的累积和的平均值
    STEP 4：对每个点将像素量化到[0,255]之间
    依赖ratio值选取而且对亮度最大区域不是白色的图像效果不佳。
    '''
    b, g, r = cv2.split(img)
    m, n, t = img.shape
    # STEP 1
