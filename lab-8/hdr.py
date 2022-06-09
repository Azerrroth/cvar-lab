#https://docs.opencv.org/3.1.0/d2/df0/tutorial_py_hdr.html
import cv2
import numpy as np

# Loading exposure images into a list
sourceDir = "./resource/lab-8/"

img_fn = [ "pic1_30.jpg", "pic1_4.jpg", "pic1.jpg", "pic4.jpg", "pic8.jpg", "pic16.jpg"]
img_list = [cv2.imread(sourceDir + fn) for fn in img_fn]
exposure_times = np.array([0.0333, 0.25, 1, 4, 8, 16], dtype=np.float32)
for item in img_list:
    item = item.resize([220, 330])
    print(item.shape)
# Merge exposures to HDR image
merge_debvec = cv2.createMergeDebevec()
print("create debevec OK!")
hdr_debvec = merge_debvec.process(img_list, times=exposure_times.copy())
print("kdr debevec OK!")
merge_robertson = cv2.createMergeRobertson()
hdr_robertson = merge_robertson.process(img_list, times=exposure_times.copy())

# Tonemap HDR image
tonemap1 = cv2.xphoto.createTonemapDurand(gamma=2.2)
res_debvec = tonemap1.process(hdr_debvec.copy())
tonemap2 = cv2.createTonemapDurand(gamma=1.3)
res_robertson = tonemap2.process(hdr_robertson.copy())

# Exposure fusion using Mertens
merge_mertens = cv2.createMergeMertens()
res_mertens = merge_mertens.process(img_list)

# Convert datatype to 8-bit and save
res_debvec_8bit = np.clip(res_debvec * 255, 0, 255).astype('uint8')
res_robertson_8bit = np.clip(res_robertson * 255, 0, 255).astype('uint8')
res_mertens_8bit = np.clip(res_mertens * 255, 0, 255).astype('uint8')

cv2.imwrite("ldr_debvec.jpg", res_debvec_8bit)
cv2.imshow("ldr_debvec", res_debvec_8bit)
cv2.imwrite("ldr_robertson.jpg", res_robertson_8bit)
cv2.imshow("ldr_robertson", res_robertson_8bit)
cv2.imwrite("fusion_mertens.jpg", res_mertens_8bit)
cv2.imshow("fusion_mertens", res_mertens_8bit)

while 1:
    key = cv2.waitKey(1)
    if key > 0:
        break
cv2.destroyAllWindows()
