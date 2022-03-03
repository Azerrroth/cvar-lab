# %%
import os
import cv2

# %%
filename = "blue.jpeg"

blued = cv2.imread(os.path.join('resource', filename))

cv2.imshow("Blue", blued)
cv2.waitKey(0)
cv2.destroyAllWindows()
