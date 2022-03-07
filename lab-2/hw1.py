# %%
from colorsys import hsv_to_rgb
import os
import cv2
import numpy as np

import wb
# %%
filename = "blue.jpeg"

blued = cv2.imread(os.path.join('resource', filename))

# cv2.imshow("Blue", blued)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

def color_correction_of_image_analysis(img):
    """
    基于图像分析的偏色检测及颜色校正方法
    :param img: cv2.imread读取的图片数据
    :return: 返回的白平衡结果图片数据
    """
  
    b, g, r = cv2.split(img)
    # print(img.shape)
    m, n = b.shape
    # detection(img)
 
    I_r_2 = np.zeros(r.shape)
    I_b_2 = np.zeros(b.shape)

    I_r_2 = (r.astype(np.float32) ** 2).astype(np.float32)
    I_b_2 = (b.astype(np.float32) ** 2).astype(np.float32)
    sum_I_r_2 = I_r_2.sum()
    sum_I_b_2 = I_b_2.sum()
    sum_I_g = g.sum()
    sum_I_r = r.sum()
    sum_I_b = b.sum()
 
    max_I_r = r.max()
    max_I_g = g.max()
    max_I_b = b.max()
    max_I_r_2 = I_r_2.max()
    max_I_b_2 = I_b_2.max()
 
    [u_b, v_b] = np.matmul(np.linalg.inv([[sum_I_b_2, sum_I_b], [max_I_b_2, max_I_b]]), [sum_I_g, max_I_g])
    [u_r, v_r] = np.matmul(np.linalg.inv([[sum_I_r_2, sum_I_r], [max_I_r_2, max_I_r]]), [sum_I_g, max_I_g])
 
    b_point = u_b * (b.astype(np.float32) ** 2) + v_b * b.astype(np.float32)
    r_point = u_r * (r.astype(np.float32) ** 2) + v_r * r.astype(np.float32)
 
    b_point[b_point > 255] = 255
    b_point[b_point < 0] = 0
    b = b_point.astype(np.uint8)
 
    r_point[r_point > 255] = 255
    r_point[r_point < 0] = 0
    r = r_point.astype(np.uint8)
 
    return cv2.merge([b, g, r])

cv2.imshow("White Balance", wb.average_white_balance(blued))

cv2.waitKey(0)
cv2.destroyAllWindows()
