import cv2
import numpy as np


def threshold(inp_image, s_thresh=(100, 255), sx_thresh=(20, 100)):

    # Make a copy of the image
    img = np.copy(inp_image)

    # Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    # Sobel x
    sobel_x = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobel_x)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    sx_binary = np.zeros_like(scaled_sobel)
    sx_binary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Stack channels (binary to colored image, S channel: blue, sobelx: green)
    color_binary = np.dstack((s_binary, sx_binary, np.zeros_like(sx_binary))) * 255

    return color_binary


def nothing(x):
    pass


# Read sample image and create window
sample_img = cv2.imread('test_images/challenge.png')
cv2.namedWindow('Set threshold values')
height, width = sample_img.shape[:2]
sample_img = cv2.resize(sample_img, (width // 2, height // 2))

# create trackbars for color change
cv2.createTrackbar('S_min', 'Set threshold values', 100, 255, nothing)
cv2.createTrackbar('S_max', 'Set threshold values', 255, 255, nothing)
cv2.createTrackbar('sx_min', 'Set threshold values', 20, 255, nothing)
cv2.createTrackbar('sx_max', 'Set threshold values', 255, 255, nothing)

# create switch for ON/OFF functionality
switch = '0 : OFF \n1 : ON'
cv2.createTrackbar(switch, 'Set threshold values', 0, 1,nothing)

# Create a copy to make changes with trackbar
thresholded = sample_img

while 1:
    k = cv2.waitKey(1) & 0xFF
    if k == 27:                 # ESC character
        break

    # get current positions of four trackbars
    s_min = cv2.getTrackbarPos('S_min', 'Set threshold values')
    s_max = cv2.getTrackbarPos('S_max', 'Set threshold values')
    sx_min = cv2.getTrackbarPos('sx_min', 'Set threshold values')
    sx_max = cv2.getTrackbarPos('sx_max', 'Set threshold values')
    s = cv2.getTrackbarPos(switch, 'Set threshold values')

    if s == 0:
        cv2.imshow('Set threshold values', sample_img)
    else:
        thresholded = threshold(sample_img, s_thresh=(s_min, s_max), sx_thresh=(sx_min, sx_max))
        cv2.imshow('Set threshold values', thresholded)

cv2.destroyAllWindows()
