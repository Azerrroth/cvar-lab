import numpy as np
import cv2

# Loading exposure images into a list
sourceDir = "./resource/lab-8/"
tar_size = (330, 220)
img_fn = [
    "pic1_30.jpg", "pic1_4.jpg", "pic1.jpg", "pic4.jpg", "pic8.jpg",
    "pic16.jpg"
]
img_list = [cv2.imread(sourceDir + fn) for fn in img_fn]
exposure_times = np.array([0.0333, 0.25, 1, 4, 8, 16], dtype=np.float32)
resized_list = []
for img in img_list:
    resized_list.append(cv2.resize(img, tar_size))

merge_debevec = cv2.createMergeDebevec()
hdr_debevec = merge_debevec.process(resized_list, times=exposure_times.copy())
merge_robertson = cv2.createMergeRobertson()
hdr_robertson = merge_robertson.process(resized_list,
                                        times=exposure_times.copy())

# Tonemap HDR image
tonemap1 = cv2.createTonemap(gamma=2.2)
res_debevec = tonemap1.process(hdr_debevec.copy())
res_robertson = tonemap1.process(hdr_robertson.copy())

# Exposure fusion using Mertens
merge_mertens = cv2.createMergeMertens()
res_mertens = merge_mertens.process(resized_list)

# Convert datatype to 8-bit and save
res_debevec_8bit = np.clip(res_debevec * 255, 0, 255).astype('uint8')
res_robertson_8bit = np.clip(res_robertson * 255, 0, 255).astype('uint8')
res_mertens_8bit = np.clip(res_mertens * 255, 0, 255).astype('uint8')
cv2.imwrite("ldr_debevec.jpg", res_debevec_8bit)
cv2.imwrite("ldr_robertson.jpg", res_robertson_8bit)
cv2.imwrite("fusion_mertens.jpg", res_mertens_8bit)

# %matplotlib inline
import matplotlib.pyplot as plt

plt.imshow(res_debevec_8bit)

# def readImagesAndTimes():
#     # 曝光时间列表
#     times = np.array([1 / 30.0, 0.25, 2.5, 15.0], dtype=np.float32)

#     # 图像文件名称列表
#     filenames = ["img_0.033.jpg", "img_0.25.jpg", "img_2.5.jpg", "img_15.jpg"]
#     images = []
#     for filename in filenames:
#         im = cv2.imread(filename)
#         images.append(im)

#     return images, times

# # 对齐输入图像
# alignMTB = cv2.createAlignMTB()
# alignMTB.process(images, images)
