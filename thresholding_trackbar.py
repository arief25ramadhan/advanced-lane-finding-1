import cv2
import numpy as np


def threshold(inp_image, mode="hls_sobelx", thresh1=(100, 255), thresh2=(20, 100)):
    # TODO: cleanup code, add select mode function
    # Make a copy of the image
    img = np.copy(inp_image)

    # Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    # Convert to Lab color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    b_channel = lab[:, :, 2]

    # Convert to LUV color space
    luv = cv2.cvtColor(img, cv2.COLOR_BGR2Luv)
    l_channel = luv[:, :, 0]

    # Sobel x
    sobel_x = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobel_x)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    sx_binary = np.zeros_like(scaled_sobel)
    sx_binary[(scaled_sobel >= thresh2[0]) & (scaled_sobel <= thresh2[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= thresh1[0]) & (s_channel <= thresh1[1])] = 1

    # Threshold color channel with b channel
    b_binary = np.zeros_like(b_channel)
    b_binary[(b_channel >= thresh2[0]) & (b_channel <= thresh2[1])] = 1

    # Threshold color channel with l channel
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= thresh1[0]) & (l_channel <= thresh1[1])] = 1

    # Stack channels (binary to colored image, S channel: blue, sobelx: green)
    color_binary = np.dstack((s_binary, sx_binary, np.zeros_like(sx_binary))) * 255

    # Stack channels (binary to colored image, S channel: blue, b channel: green)
    color_binary_lab = np.dstack((s_binary, b_binary, np.zeros_like(b_binary))) * 255

    # Stack channels (binary to colored image, L channel: blue, b channel: green)
    color_binary_labluv = np.dstack((l_binary, b_binary, np.zeros_like(b_binary))) * 255

    return color_binary_labluv


def nothing(x):
    pass


# Read sample image and create window
sample_img = cv2.imread('test_images/challenge.png')
cv2.namedWindow('Set threshold values')
height, width = sample_img.shape[:2]
sample_img = cv2.resize(sample_img, (width // 2, height // 2))

# create trackbars for color change
cv2.createTrackbar('L_min', 'Set threshold values', 185, 255, nothing)
cv2.createTrackbar('L_max', 'Set threshold values', 255, 255, nothing)
cv2.createTrackbar('b_min', 'Set threshold values', 140, 255, nothing)
cv2.createTrackbar('b_max', 'Set threshold values', 200, 255, nothing)

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
    s_min = cv2.getTrackbarPos('L_min', 'Set threshold values')
    s_max = cv2.getTrackbarPos('L_max', 'Set threshold values')
    sx_min = cv2.getTrackbarPos('b_min', 'Set threshold values')
    sx_max = cv2.getTrackbarPos('b_max', 'Set threshold values')
    s = cv2.getTrackbarPos(switch, 'Set threshold values')

    if s == 0:
        cv2.imshow('Set threshold values', sample_img)
    else:
        thresholded = threshold(sample_img, thresh1=(s_min, s_max), thresh2=(sx_min, sx_max))
        cv2.imshow('Set threshold values', thresholded)

cv2.destroyAllWindows()
