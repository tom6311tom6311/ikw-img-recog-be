import cv2
import math
import numpy as np
from utils.img_utils import color_interpolation, create_circular_mask, create_rectangular_mask, transform, heatmap

STREAM_URL = 'data/video/IMG_6541.MOV'
TARGET_AREA_HEIGHT = 50 # in percentage of full height
TARGET_AREA_WIDTH = 50 # in percentage of full width
TARGET_AREA_LEFT = 25 # in percentage of full width
TARGET_AREA_TOP = 25 # in percentage of full height

vcap = cv2.VideoCapture(STREAM_URL)

if (vcap.isOpened() == False):
  print("Error opening video stream")

ret, frame = vcap.read()
height, width, _ = frame.shape
mask_height = round(TARGET_AREA_HEIGHT / 100 * height)
mask_width = round(TARGET_AREA_WIDTH / 100 * width)
mask_left = round(TARGET_AREA_LEFT / 100 * width)
mask_top = round(TARGET_AREA_TOP / 100 * height)
mask = create_rectangular_mask(height, width, mask_left, mask_top, mask_width, mask_height)

score_min = 1000000000
score_max = 0
while (vcap.isOpened()):
  ret, frame = vcap.read()
  if (ret == True):
    frame_target = frame[mask_top: mask_top + mask_height, mask_left: mask_left + mask_width, :]
    frame_target_transformed = transform(frame_target)
    # frame_masked = transform(frame) * mask
    score = np.sum(frame_target_transformed) / 255
    if (score < score_min):
      score_min = score - 1
    if (score > score_max):
      score_max = score
    ring_color = color_interpolation((255, 0, 0), (0, 0 ,255), (score - score_min) / (score_max - score_min))
    # frame_masked = heatmap(frame_masked)
    # frame_processed = frame_masked | frame
    frame[mask_top: mask_top + mask_height, mask_left: mask_left + mask_width, :] = np.stack((frame_target_transformed,)*3, axis=-1)
    cv2.rectangle(frame, (mask_left, mask_top), (mask_left + mask_width, mask_top + mask_height), ring_color, 5)
    cv2.imshow('Frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
  else:
    break

vcap.release()

cv2.destroyAllWindows()
