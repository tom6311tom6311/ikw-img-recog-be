import cv2
import math
import numpy as np
from matplotlib import pyplot as plt

# STREAM_URL = 'rtsp://tom6311tom6311:qa1968qa@192.168.0.182:88/videoMain'
# STREAM_URL = 'http://122.116.122.250/cgi-bin/hi3510/snap.cgi?&-getstream&-chn=2'
STREAM_URL = 'data/video/IMG_6540.MOV'
FRAME_PATH = 'data/capture.jpg'
TARGET_AREA = [50, 100, 40, 70] # [left, right, top, bottom] in percentage

vcap = cv2.VideoCapture(STREAM_URL)

# ret, frame = vcap.read()
ret, frame1 = vcap.read()

vcap.set(1, 30)

ret, frame2 = vcap.read()
frame = frame1 - frame2

cv2.imwrite(FRAME_PATH, frame)

# frame = cv2.imread('data/test.jpg')
frame = cv2.imread(FRAME_PATH)
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
height, width = frame.shape
# frame = frame[math.floor(height * TARGET_AREA[2] / 100):math.floor(height * TARGET_AREA[3] / 100), math.floor(width * TARGET_AREA[0] / 100):math.floor(width * TARGET_AREA[1] / 100)]

# Blurring
# frame = cv2.medianBlur(frame, 3)

# Laplacian
binarized = cv2.Laplacian(frame, cv2.CV_8U, ksize=1)

# Static Thresholding
# ret,binarized = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)

# Adaptive Mean Thresholding
# binarized = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

# Adaptive Gaussian Thresholding
# binarized = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# Do averaging convolution
kernel = np.ones((5,5))
convolved = cv2.filter2D(binarized, -1, kernel)
# ret,convolved = cv2.threshold(convolved, 225, 255, cv2.THRESH_BINARY)

heatmap = None
heatmap = cv2.normalize(255 - convolved, heatmap, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# Prepare for drawing
titles = ['Grayscale Image', 'Gradient', 'Convolved', 'Heatmap']
images = [frame, binarized, convolved, heatmap]

# Plotting results
for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()
