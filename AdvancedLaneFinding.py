import numpy as np
import cv2


# LINE CLASS for storing values

class Line:
    def __init__(self):
        # Was line detected in the previous frame?
        self.detected = False
        # Recent fits
        self.previous_fits = []
        # Polynomial coefficients averaged
        self.best_fit = None
        # X values plotted
        self.current_fitx = None
        # Polynomial coefficients
        self.current_fit = np.array([0, 0, 0])    # back to 0, 0, 0 -> False ?
        # Polynomial coefficients
        self.previous_fit = np.array([0, 0, 0])
        # Radius of curvature
        self.radius_of_curvature = 1000
        # Position of vehicle (dist from center)
        self.line_base_pos = None
        # Difference in coefficients
        self.diffs = np.array([0, 0, 0], dtype='float')
        # For first frame
        self.initialized = False
        # Average fit
        self.average_fit = np.array([0, 0, 0])
        # Average curvature
        self.average_curvature = 0
        # Previous curves
        self.previous_curves = []
        # Frame counter
        self.frame_cnt = 0


class Lane:
    def __init__(self):
        self.bottom_width = 0
        self.top_width = 0
        self.average_bottom_width = 0
        self.average_top_width = 0
        self.previous_bottom_widths = []
        self.previous_top_widths = []


# FUNCTION DEFINITIONS


def camera_calibration(image_files, nx, ny):
    # Expects a list of image files
    # Prepare grid for object points
    objp = np.zeros((ny*nx, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    # Create arrays for storing object and image points
    objpoints = []
    imgpoints = []

    # Loop through images, append image and object points
    for fname in image_files:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        if ret is True:
            # Add object and image points to the correesponding arrays
            objpoints.append(objp)
            imgpoints.append(corners)

    # Get camera matrix and distortion coefficients
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return mtx, dist


def undistort(img, mtx, dist):

    # Undistort image using camera matrix & distortion coefficients
    return cv2.undistort(img, mtx, dist)


def threshold(img, l_thresh=(185, 255), b_thresh=(140, 200), sx_thresh=(10, 100)):

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
    # TODO: implement yellow AND white mask

    # Make a copy of the image
    img = np.copy(img)

    # Convert to Lab color space
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    b_channel = lab[:, :, 2]

    # Convert to LUV color space
    luv = cv2.cvtColor(img, cv2.COLOR_RGB2Luv)
    l_channel = luv[:, :, 0]

    # Threshold b color channel
    b_binary = np.zeros_like(b_channel)
    b_binary[(b_channel >= b_thresh[0]) & (b_channel <= b_thresh[1])] = 1

    # Threshold l color channel
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])] = 1

    # Sobelx
    sobel_x = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobel_x)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    sx_binary = np.zeros_like(scaled_sobel)
    sx_binary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Sobelx AND white
    sobel_white_binary = np.zeros_like(l_channel)
    sobel_white_binary[(sx_binary == 1) & (l_binary == 1)] = 1

    # Sobelx AND yellow
    sobel_yellow_binary = np.zeros_like(l_channel)
    sobel_yellow_binary[(sx_binary == 1) & (b_binary == 1)] = 1

    white_sobelx_and_color = np.dstack(
        (sobel_white_binary, sobel_yellow_binary, np.zeros_like(sobel_white_binary))) * 255

    # Combine sobelx-yellow and sobelx-white
    combined_binary_sobel = np.zeros_like(b_binary)
    combined_binary_sobel[(sobel_white_binary == 1) | (sobel_yellow_binary == 1)] = 1

    # Stack channels (binary to colored image, L channel: blue, b channel: green)
    color_binary = np.dstack((np.zeros_like(l_binary), b_binary, l_binary)) * 255

    # Combine thresholds
    combined_binary = np.zeros_like(b_binary)
    combined_binary[(b_binary == 1) | (l_binary == 1)] = 1
    combined_binary_out = np.dstack((combined_binary, combined_binary, combined_binary)) * 255

    # return color_binary
    # cv2.imwrite('output_images/02_thresholded.jpg', color_binary)
    return combined_binary_sobel, white_sobelx_and_color


def perspective_tr(img):

    # TODO: check for any unintended horizontal shift

    img_size = (img.shape[1], img.shape[0])

    # Define source and destination points based on a straight road section
    src_pts = np.float32([[595, 450], [688, 450], [220, 720], [1105, 720]])
    dst_pts = np.float32([[300, 0], [900, 0], [300, 720], [900, 720]])

    # Calculate transform matrix and inverse transform matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    Minv = cv2.getPerspectiveTransform(dst_pts, src_pts)
    transformed = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return transformed, M, Minv


def fit_poly(img_shape, leftx, lefty, rightx, righty):

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return left_fit, right_fit


def sliding_windows(img):
    # TODO: In cases where no peaks are found, we place a window centered at the location calculated assuming
    # the location of previous window moved by a precomputed offset.

    # Calculate histogram by summing bottom half of image
    bottom_half = img[img.shape[0] // 2:, :]
    histogram = np.sum(bottom_half, axis=0)

    # Output image for testing
    out_img = np.dstack((img, img, img))

    # Create starting point on left and right side and set them as current points
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Parameters of sliding window
    # Number of windows
    nwindows = 20
    # Width of windows
    margin = 80
    # Minimum number of pixels to recenter window
    minpix = 50

    # Set window height
    window_height = np.int(img.shape[0] // nwindows)

    # Find nonzero pixels
    nonzero = img.nonzero()
    nonzerox = np.array(nonzero[1])
    nonzeroy = np.array(nonzero[0])

    # Empty lists for storing lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through windows
    for window in range(nwindows):
        # Window boundaries:
        win_y_low = img.shape[0] - (window + 1) * window_height
        win_y_high = img.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Identify pixels within the windows
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & \
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & \
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Add the found pixels to lane line
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # Update x axis position based on pixels found
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate list of pixels
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        pass

    # Get left and right lane pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Color left and right line pixels
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Fit polynomial based on pixels found
    left_fit, right_fit = fit_poly(img.shape, leftx, lefty, rightx, righty)

    left_fit_text = "left: %.6f %.6f %.6f" % (left_fit[0], left_fit[1], left_fit[2])
    right_fit_text = "right: %.6f %.6f %.6f" % (right_fit[0], right_fit[1], right_fit[2])

    # Add text to image
    cv2.putText(out_img, left_fit_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), thickness=2)
    cv2.putText(out_img, right_fit_text, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), thickness=2)

    return left_fit, right_fit, out_img


def search_around_poly(binary_warped, left_fit, right_fit):

    # Margin for searching around curve
    margin = 60

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Mask to get non-zero pixels that are next to the curve within margin
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                    left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                    left_fit[1] * nonzeroy + left_fit[2] + margin)))

    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                    right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                    right_fit[1] * nonzeroy + right_fit[2] + margin)))

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Generate y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])

    # Fit polynomial based on pixels found
    left_fit, right_fit = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)

    left_fitx, right_fitx = calc_x_values(binary_warped.shape, left_fit, right_fit)

    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    window_img = np.zeros_like(out_img)

    # Color left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))

    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    left_fit_text = "left: %.6f %.6f %.6f" % (left_fit[0], left_fit[1], left_fit[2])
    right_fit_text = "right: %.6f %.6f %.6f" % (right_fit[0], right_fit[1], right_fit[2])

    # Add text to image
    cv2.putText(out_img, left_fit_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), thickness=2)
    cv2.putText(out_img, right_fit_text, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), thickness=2)

    return left_fit, right_fit, out_img


def measure_curvature_real(img_shape, fit):
    # Calculates the curvature of polynomial functions in meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    # Generate y values for plotting
    ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])

    # Calculate x values using polynomial coeffs
    fitx = fit[0] * ploty ** 2 + fit[1] * ploty + fit[2]

    # Evaluate at bottom of image
    y_eval = np.max(ploty)

    # Fit curves with corrected axes
    curve_fit = np.polyfit(ploty * ym_per_pix, fitx * xm_per_pix, 2)

    # Calculate curvature values for left and right lanes
    curvature = ((1 + (2 * curve_fit[0] * y_eval * ym_per_pix + curve_fit[1]) ** 2) ** 1.5) / np.absolute(
        2 * curve_fit[0])

    return curvature


def get_position(img_shape, left_lane_pos, right_lane_pos):

    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    # Calculate position based on midpoint - center of lanes distance
    midpoint = img_shape // 2
    center_of_lanes = (right_lane_pos + left_lane_pos) / 2
    position = midpoint - center_of_lanes

    # Get value in meters
    real_position = position * xm_per_pix

    return real_position


def check_fit(left_lane, right_lane, lane):
    # Difference in average and current coefficients:
    coeff_diff_right = np.sum((right_lane.current_fit[0] - right_lane.previous_fit[0])**2)
    coeff_diff_right = np.sqrt(coeff_diff_right)

    coeff_diff_left = np.sum((left_lane.current_fit[0] - left_lane.previous_fit[0]) ** 2)
    coeff_diff_left = np.sqrt(coeff_diff_left)

    curve_diff_right = right_lane.radius_of_curvature - right_lane.average_curvature
    curve_diff_left = left_lane.radius_of_curvature - left_lane.average_curvature
    curve_diff = abs(right_lane.radius_of_curvature - left_lane.radius_of_curvature)

    top_width_diff = lane.top_width - lane.average_top_width
    bottom_width_diff = lane.bottom_width - lane.average_bottom_width

    width_check_top = top_width_diff > 0.5 * lane.average_top_width or lane.top_width > 1.25 * lane.bottom_width
    width_check_bottom = bottom_width_diff > 0.5 * lane.average_bottom_width
    # print("Average bottom: ", lane.average_bottom_width)
    # print("Average top: ", lane.average_top_width)
    # print("Previous bottom: ", lane.previous_bottom_widths)
    # print("Previous top: ", lane.previous_top_widths)
    # print("Current bottom: ", lane.bottom_width)
    # print("Current top: ", lane.top_width)

    # Check if parameters are ok
    if (left_lane.initialized is True) and (right_lane.initialized is True):
        if (left_lane.frame_cnt > 1) and (right_lane.frame_cnt > 1):
            if abs(curve_diff_left) > 0.5 * left_lane.average_curvature or \
                    abs(curve_diff_right) > 0.5 * right_lane.average_curvature or \
                    curve_diff > 0.5 * left_lane.average_curvature or \
                    curve_diff > 0.5 * right_lane.average_curvature or \
                    width_check_bottom or width_check_top:
                result = False
            else:
                result = True
        else:
            result = True
    else:
        result = True

    return result


def average_fits(img_shape, lane):
    #TODO: make this more sophisticated:
    # right_fit = .05 * right_fit + .95 * right_fit_prev ?
    # Calculates the average fit based on previous n values
    # TODO: set average as global var
    sum = 0
    n = 5
    average_fit = [0, 0, 0]

    # TODO: check if these can be done in the previous ifs
    # If we do not have enough fits, append the list with the current fit
    if len(lane.previous_fits) < n:
        lane.previous_fits.append(lane.current_fit)
    # If amount of fits == n, remove the last element and add the current one
    if len(lane.previous_fits) == n:
        lane.previous_fits.pop(n-1)
        lane.previous_fits.insert(0, lane.current_fit)

    # If we have enough fits, calculate the average
    # TODO: sort these fors in some logical order...
    if (len(lane.previous_fits) > 0):
        for i in range(0, 3):
            for num in range(0, len(lane.previous_fits)):

                sum = sum + lane.previous_fits[num][i]
                # TODO: divide by length instead, checking for zero division
            average_fit[i] = sum / len(lane.previous_fits)

    return average_fit


def average_curvature(img_shape, lane):
    sum = 0
    n = 5
    average_curve = 0

    if len(lane.previous_curves) < n:
        lane.previous_curves.append(lane.radius_of_curvature)
    # If amount of fits == n, remove the last element and add the current one
    if len(lane.previous_curves) == n:
        lane.previous_curves.pop(n-1)
        lane.previous_curves.insert(0, lane.radius_of_curvature)

    # If we have enough fits, calculate the average
    # TODO: sort these fors in some logical order...
    if (len(lane.previous_curves) > 0):
        for i in range(0, len(lane.previous_curves)):

            sum = sum + lane.previous_curves[i]
                # TODO: divide by length instead, checking for zero division
        average_curve = sum / len(lane.previous_curves)

    return average_curve


def average_width(img_shape, lane):
    sum_bottom = 0
    sum_top = 0
    n = 5
    average_bottom_width = 0
    average_top_width = 0

    if len(lane.previous_bottom_widths) < n:
        lane.previous_bottom_widths.append(lane.bottom_width)
    # If amount of fits == n, remove the last element and add the current one
    if len(lane.previous_bottom_widths) == n:
        lane.previous_bottom_widths.pop(n-1)
        lane.previous_bottom_widths.insert(0, lane.bottom_width)

    # If we have enough fits, calculate the average
    # TODO: sort these fors in some logical order...
    if (len(lane.previous_bottom_widths) > 0):
        for i in range(0, len(lane.previous_bottom_widths)):

            sum_bottom = sum_bottom + lane.previous_bottom_widths[i]
            average_bottom_width = sum_bottom / len(lane.previous_bottom_widths)


    if len(lane.previous_top_widths) < n:
        lane.previous_top_widths.append(lane.top_width)
    # If amount of fits == n, remove the last element and add the current one
    if len(lane.previous_top_widths) == n:
        lane.previous_top_widths.pop(n-1)
        lane.previous_top_widths.insert(0, lane.top_width)

    # If we have enough fits, calculate the average
    # TODO: sort these fors in some logical order...
    if (len(lane.previous_top_widths) > 0):
        for i in range(0, len(lane.previous_top_widths)):

            sum_top = sum_top + lane.previous_top_widths[i]
            average_top_width = sum_top / len(lane.previous_top_widths)

    return average_bottom_width, average_top_width


def draw_lanes(warped, undist, left_fit, right_fit, curvature, position, Minv):

    # Generate y values
    ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])

    # Calculate x values
    left_fitx, right_fitx = calc_x_values(warped.shape, left_fit, right_fit)

    # Create image to draw lines onto
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))

    # Combine the result with the original image
    lanes = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    # Texts to write on image
    curv_text = "Curvature: %.2f meters" % curvature
    pos_text = "Position: %.2f from center" % position

    # Add text to image
    cv2.putText(lanes, curv_text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
    cv2.putText(lanes, pos_text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

    return lanes


def calc_x_values(img_shape, left_fit, right_fit):
    # Generate y values for plotting
    ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])

    # Calculate x values using polynomial coeffs
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    return left_fitx, right_fitx