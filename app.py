import numpy as np
import matplotlib.pyplot as plt
import cv2

image = cv2.imread('images/boardwalk.jpg')
image_copy = cv2.imread('images/boardwalk.jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
grayscale = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)

# Reshape to 3-dimensional matrix
pixel_arr_colored = image.reshape((-1,3))
pixel_arr_colored = np.float32(pixel_arr_colored)

pixel_arr_gray = grayscale.reshape((-1, 1))
pixel_arr_gray = np.float32(pixel_arr_gray)

# Criteria for stopping k-means algo
criteria_colored = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.75)
criteria_gray = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)

k = 10 # number of clusters
attempts = 10
retval, labels, centers = cv2.kmeans(pixel_arr_colored, k, None, criteria_colored, attempts, cv2.KMEANS_RANDOM_CENTERS)
retvalg, labelsg, centersg = cv2.kmeans(pixel_arr_gray, k, None, criteria_gray, attempts, cv2.KMEANS_RANDOM_CENTERS)

# Cluster centers
centers = np.uint8(centers)
centersg = np.uint8(centersg)

seg = centers[labels.flatten()]
seg_g = centersg[labelsg.flatten()]

seg_img = seg.reshape(image.shape)
seg_img_gray = seg_g.reshape(grayscale.shape)

# plt.figure(figsize=(15, 12))
# plt.subplot(1, 2, 1)
plt.imshow(seg_img)
plt.title('Generated Colour Image when k = %i' % k)
# plt.subplot(1, 2, 2)
# plt.imshow(seg_img_gray)
# plt.title('Greyscaled Generated Image for k = %i' % k)
plt.show()