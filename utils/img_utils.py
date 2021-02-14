import cv2
import numpy as np

SOBEL_X_KERNEL = np.array([[-1, 0, 1],
                          [-2, 0, 2],
                          [-1, 0, 1]])
SOBEL_Y_KERNEL = np.array([[-1, -2, -1],
                          [0, 0, 0],
                          [1, 2, 1]])
SOBEL_D1_KERNEL = np.array([[0,1,2],
                          [-1,0,1],
                          [-2,-1,0]])
SOBEL_D2_KERNEL = np.array([[-2,-1,0],
                          [-1,0,1],
                          [0,1,2]])
BLUR_KERNEL = np.ones((3,3))

def color_interpolation(color1, color2, ratio):
  return (
    round(color1[0] + (color2[0] - color1[0]) * ratio),
    round(color1[1] + (color2[1] - color1[1]) * ratio),
    round(color1[2] + (color2[2] - color1[2]) * ratio)
  )

def create_circular_mask(full_height, full_width, center, radius):
  Y, X = np.ogrid[:full_height, :full_width]
  dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
  mask = dist_from_center <= radius
  return mask

def create_rectangular_mask(full_height, full_width, left, top, width, height):
  mask = np.zeros((full_height, full_width))
  mask[left: left + width, top: top + height] = 1
  return mask

def highpass(frame_2d):
  f = np.fft.fft2(frame_2d)
  fshift = np.fft.fftshift(f)
  rows, cols = frame_2d.shape
  crow, ccol = rows//2 , cols//2
  fshift[crow - 200: crow + 201, ccol - 200: ccol + 201] = 0
  f_ishift = np.fft.ifftshift(fshift)
  frame_2d_back = np.fft.ifft2(f_ishift)
  frame_2d_back = np.real(frame_2d_back)
  return frame_2d_back

def transform(frame):
  grayed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  # gradient = cv2.Laplacian(grayed, cv2.CV_16U, ksize=1)
  edge_detected_x = cv2.filter2D(grayed, -1, SOBEL_X_KERNEL)
  edge_detected_y = cv2.filter2D(grayed, -1, SOBEL_Y_KERNEL)
  # edge_detected_d1 = cv2.filter2D(grayed, -1, SOBEL_D1_KERNEL)
  # edge_detected_d2 = cv2.filter2D(grayed, -1, SOBEL_D2_KERNEL)
  edge_detected = np.minimum(edge_detected_x, edge_detected_y)
  blurred = cv2.GaussianBlur(edge_detected, (9,9), 0)
  # highpassed = highpass(grayed)
  # canny = cv2.Canny(grayed, 200, 300)
  _, binarized = cv2.threshold(blurred, 2, 255, cv2.THRESH_BINARY)
  # print(grayed.shape)
  return binarized

def heatmap(frame_2d):
  heatmap = None
  heatmap = cv2.normalize(frame_2d, heatmap, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
  heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
  return heatmap