import cv2
import numpy as np
import AdvancedLaneFinding as alf

def threshold(inp_image, s_thresh=(100, 255), sx_thresh=(20, 100)):
    # TODO: create trackbar GUI with OpenCV

    # TODO : check other color spaces
    # If you want to continue to explore additional color channels, I have seen that the L channel from LUV with lower
    # and upper thresholds around 225 & 255 respectively works very well to pick out the white lines, even in the parts
    # of the video with heavy shadows and brighter pavement. You can also try out the b channel from Lab which does a
    # great job with the yellow lines (you can play around with thresholds around 155 & 200).

    # TODO: color thresholding in all colorspaces
    # You could try color thresholding in all RGB, HLS, HSV colorspaces to make the pipeline more robust. Color
    # thresholding is also much faster to compute as opposed to the gradient calculation in the Sobel transform.

    # Lab is another colorspace that should work well here, especially the "B" channel which should help identify the
    # yellow lanes effectively.

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

    # Combine thresholds
    combined_binary = np.zeros_like(sx_binary)
    combined_binary[(s_binary == 1) | (sx_binary == 1)] = 1
    combined_binary_out = np.dstack((combined_binary, combined_binary, combined_binary)) * 255
    cv2.imwrite('output_images/02_thresholded.jpg', color_binary)
    img = color_binary
    return img

    # return combined_binary


def nothing(x):
    pass


# Create a black image, a window
img = cv2.imread('test_images/challenge.png')
cv2.namedWindow('image')
height, width = img.shape[:2]
img = cv2.resize(img, (width // 2, height // 2))

# create trackbars for color change
cv2.createTrackbar('S_min', 'image', 100, 255, nothing)
cv2.createTrackbar('S_max', 'image', 255, 255, nothing)
cv2.createTrackbar('sx_min', 'image', 20, 255, nothing)
cv2.createTrackbar('sx_max', 'image', 255, 255, nothing)

# create switch for ON/OFF functionality
switch = '0 : OFF \n1 : ON'
cv2.createTrackbar(switch, 'image', 0, 1,nothing)

# Create a copy
thresholded = img

while(1):
    #cv2.imshow('image', thresholded)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:     # ESX character
        break

    # get current positions of four trackbars
    s_min = cv2.getTrackbarPos('S_min', 'image')
    s_max = cv2.getTrackbarPos('S_max','image')
    sx_min = cv2.getTrackbarPos('sx_min','image')
    sx_max = cv2.getTrackbarPos('sx_max','image')
    s = cv2.getTrackbarPos(switch,'image')

    if s == 0:
        cv2.imshow('image', img)
    else:
        thresholded = threshold(img, s_thresh=(s_min, s_max), sx_thresh=(sx_min, sx_max))
        cv2.imshow('image', thresholded)

cv2.destroyAllWindows()
