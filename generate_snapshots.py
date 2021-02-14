# Generate snapshots given a video
# Usage: python3 generate_snapshots.py data/video/<xxx.MOV> data/snapshots/<xxx>

import sys
import os
import shutil
import cv2

INPUT_VIDEO_PATH = sys.argv[1]
OUTPUT_FOLDER_PATH = sys.argv[2]
INTERVAL = 30 # seconds

if os.path.exists(OUTPUT_FOLDER_PATH):
  shutil.rmtree(OUTPUT_FOLDER_PATH)

os.makedirs(OUTPUT_FOLDER_PATH)

vcap = cv2.VideoCapture(INPUT_VIDEO_PATH)
fps = round(vcap.get(cv2.CAP_PROP_FPS))
multiplier = fps * INTERVAL

current_frame = 0
while (vcap.isOpened()):
  ret, frame = vcap.read()
  if (ret == True):
    if current_frame % multiplier == 0:
      cv2.imwrite(f'{OUTPUT_FOLDER_PATH}/frame_{current_frame}.jpg', frame)
    current_frame += 1
  else:
    break

# When everything done, release the capture
vcap.release()
cv2.destroyAllWindows()