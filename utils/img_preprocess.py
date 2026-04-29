import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

"""
cv.namedWindow('im_win', cv.WINDOW_NORMAL)
cv.resizeWindow('im_win', 800, 600)
"""

# gray scale image
img = cv.imread('./img/raw/raw_008.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray = gaussian_blur = cv.GaussianBlur(gray, (5, 5), 0)

# CLAHE (https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html#autotoc_md1386)
clahe = cv.createCLAHE(clipLimit=5.0, tileGridSize=(10,10))
cl1 = clahe.apply(gray)

#Thresholding
img_thresh = cv.adaptiveThreshold(cl1, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 31, 0)

#Morphological Closing
kernel_size = 3  # Increase from 1 to connect segments better
kernel = cv.getStructuringElement(cv.MORPH_RECT, (kernel_size, kernel_size))

# First dilate to connect broken segments
dilated = cv.dilate(img_thresh, kernel, iterations=2)

# Then erode to restore size
eroded = cv.erode(dilated, kernel, iterations=2)

# Final closing for smoothing
closed = cv.morphologyEx(eroded, cv.MORPH_CLOSE, kernel, iterations=1)


# image display
"""
cv.imshow('Gray Image', gray)
cv.imshow('Canny Edges', edges)
cv.waitKey(0)
cv.destroyAllWindows()
"""

plt.subplot(1, 4, 1)
plt.title('Original Image')
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 4, 2)
plt.title('Gray Scale Image')
plt.imshow(gray, cmap='gray')
plt.axis('off') 

plt.subplot(1, 4, 3)
plt.title('CLAHE Image')
plt.imshow(cl1, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.title('Morphological Closing')
plt.imshow(closed, cmap='gray')
plt.axis('off')

plt.show()