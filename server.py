import os
import math
import random
import cv2
import numpy as np
from flask import Flask, send_file, request
from flask_cors import CORS
from glob import glob
from utils.img_utils import transform

app = Flask(__name__)
CORS(app)

@app.route('/snapshot')
def get_snapshot():
  try:
    path = request.args.get('path')
    if (not path):
      return send_file(f'data/snapshots/default.jpg')
    else:
      return send_file(f'data/snapshots/{path}')
  except Exception as e:
    print(str(e))
    return send_file(f'data/snapshots/default.jpg')

@app.route('/detection')
def get_detection():
  try:
    all_snapshots = [y for x in os.walk('data/snapshots') for y in glob(os.path.join(x[0], '*.jpg'))]
    current_snapshot = all_snapshots[math.floor(random.random() * len(all_snapshots))]
    image = cv2.imread(current_snapshot)
    height, width, _ = image.shape
    mask_height = round((int(request.args.get('height')) or 50) / 100 * height)
    mask_width = round((int(request.args.get('width')) or 50) / 100 * width)
    mask_left = round((int(request.args.get('left')) or 25) / 100 * width)
    mask_top = round((int(request.args.get('top')) or 25) / 100 * height)
    if (mask_height == 0 or mask_width == 0):
      return { 'score': 0, 'path': current_snapshot[15:] }
    target = image[mask_top: mask_top + mask_height, mask_left: mask_left + mask_width, :]
    target_transformed = transform(target)
    cv2.imwrite('test.jpg', target_transformed)
    return { 'score': round(np.sum(target_transformed) / 255 / (mask_width * mask_height) * 100), 'path': current_snapshot[15:] }
  except Exception as e:
    return str(e)

if __name__ == '__main__':
  app.run(debug=True, host='0.0.0.0')