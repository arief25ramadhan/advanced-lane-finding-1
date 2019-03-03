import numpy as np
import cv2
import glob
import datetime
import AdvancedLaneFinding as alf
from moviepy.editor import VideoFileClip

# STUFF FOR TESTING


def test_pipeline(mode):
    # Select test mode
    if mode is "image":
        print("Testing on images")
        image_pipeline()

    elif mode is "video1":
        print("Testing first project video")
        video_pipeline(1)

    elif mode is "video2":
        print("Testing challenge video")
        video_pipeline(2)

    elif mode is "video3":
        print("Testing extra hard challenge video. Good luck...")
        video_pipeline(3)
    else:
        print("Error! Mode must be: image/video1/video2/video3")


def image_pipeline():
    # APPLY PIPELINE ON IMAGE

    # Code snippet for optimizing for a parameter:
    # for color_thr_max in range(0, 255, 25):
        #test_image = cv2.imread('test_images/test4.jpg')
        #result = find_lane_lines(test_image, (100, color_thr_max), (20, 100))
        # Save output images
        #output_fname = 'output_images/test_output_color_thr_100_'
        #file_num = color_thr_max
        #cv2.imwrite(output_fname + str(file_num) + '.jpg', result)


    for num in range(3,4):
        test_image = cv2.imread('test_images/test' + str(num) + '.jpg')
        result = find_lane_lines(test_image)
        # Save output images
        output_fname_image = 'output_images/test_output'
        cv2.imwrite(output_fname_image + str(num) + '.jpg', result)


def video_pipeline(video, mode="long"):
    # APPLY PIPELINE ON VIDEO

    # Select input
    if video is 1:
        filename = 'project_video'
    elif video is 2:
        filename = 'challenge_video'
    elif video is 3:
        filename = 'harder_challenge_video'

    # Make only short subclip:
    if mode is "long":
        test_input = VideoFileClip(filename + '.mp4')
    elif mode is "short":
        test_input = VideoFileClip(filename + '.mp4').subclip(0, 3)

    # Name ouput file
    date = datetime.datetime.now().strftime("_%Y_%m_%d_%H_%M")
    output_fname_video = 'output_videos/output_' + filename + date +'.mp4'

    # Process input video, write to output file
    test_output = test_input.fl_image(find_lane_lines)
    test_output.write_videofile(output_fname_video, audio=False)


# PIPELINE FOR ADVANCED LANE FINDING

def find_lane_lines(img):

    # 1) Apply distortion correction
    undistorted = alf.undistort(img, cam_mtx, dist_coeffs)

    # 2) Use color transforms, gradients, etc., to create a thresholded binary image.
    thresholded, colored_thresholded = alf.threshold(undistorted)

    # 3) Apply perspective transform
    top_view, M, Minv = alf.perspective_tr(thresholded)

    # 4) Detect lane pixels and fit polynomial
    # If previous lane was detected, search next to curve, otherwise use window method
    if (left_lane.detected is False) or (right_lane.detected is False):
        try:
            left_fit, right_fit, lanes_colored = alf.sliding_windows(top_view)
        except TypeError:       # if nothing was found, use previous fit
            left_fit = left_lane.previous_fit
            right_fit = right_lane.previous_fit
            lanes_colored = np.zeros_like(img)
    else:
        try:
            left_fit, right_fit, lanes_colored = alf.search_around_poly(top_view, left_lane.previous_fit, right_lane.previous_fit)
        except TypeError:
            try:
                left_fit, right_fit, lanes_colored = alf.sliding_windows(top_view)
            except TypeError:  # if nothing was found, use previous fit
                left_fit = left_lane.previous_fit
                right_fit = right_lane.previous_fit
                lanes_colored = np.zeros_like(img)

    left_lane.current_fit = left_fit
    right_lane.current_fit = right_fit

    # TODO: make img_shape a global constant?
    # TODO: initialize left_fit, right_fit to some thing

    # Calculate base position of lane lines to get lane distance
    left_lane.line_base_pos = left_fit[0] * (top_view.shape[0] - 1) ** 2 + left_fit[1] * (top_view.shape[0] - 1) + left_fit[2]
    right_lane.line_base_pos = right_fit[0] * (top_view.shape[0] - 1) ** 2 + right_fit[1] * (top_view.shape[0] - 1) + right_fit[2]
    lane_distance = right_lane.line_base_pos - left_lane.line_base_pos

    # 5) Determine lane curvature and position of the vehicle
    # Calculate curvature
    left_lane.radius_of_curvature = alf.measure_curvature_real(top_view.shape, left_fit)
    right_lane.radius_of_curvature = alf.measure_curvature_real(top_view.shape, right_fit)

    # Take the mean value of the two curvatures
    curvature = left_lane.radius_of_curvature + right_lane.radius_of_curvature / 2

    # Calculate vehicle position
    vehicle_position = alf.get_position(top_view.shape[1], left_lane.line_base_pos, right_lane.line_base_pos)

    # Check if values make sense
    #if left_lane.detected and right_lane.detected is True:
    if alf.check_fit(left_lane, right_lane) is False:
        # TODO: dont set previous fit if its the first frame
        # If fit is not good, use previous values and indicate that lanes were not found
        left_lane.current_fit = left_lane.previous_fit
        right_lane.current_fit = right_lane.previous_fit
        left_lane.detected = False
        right_lane.detected = False

    else:
        # If fit is good, use current values and indicate that lanes were found
        left_lane.current_fit = left_fit
        right_lane.current_fit = right_fit
        left_lane.detected = True
        right_lane.detected = True
        left_lane.initialized = True
        right_lane.initialized = True
        left_lane.frame_cnt += 1
        right_lane.frame_cnt += 1

    # Calculate the average of the recent fits and set this as the current fit
    left_lane.average_fit = alf.average_fits(top_view.shape, left_lane)
    right_lane.average_fit = alf.average_fits(top_view.shape, right_lane)
    left_lane.average_curvature = alf.average_curvature(top_view.shape, left_lane)
    right_lane.average_curvature = alf.average_curvature(top_view.shape, right_lane)

    # Set average value as current value
    #left_lane.current_fit = left_lane.average_fit
    #right_lane.current_fit = right_lane.average_fit

    # Update all calculations based on averaged values
    #curvature = alf.measure_curvature_real(top_view.shape, left_lane.current_fit, right_lane.current_fit)
    #left_lane.radius_of_curvature = curvature
    #right_lane.radius_of_curvature = curvature

    #left_lane.line_base_pos = average_left_fitx[top_view.shape[0]-1]
    #right_lane.line_base_pos = average_right_fitx[top_view.shape[0]-1]
    #vehicle_position = alf.get_position(undistorted.shape[1], left_lane.line_base_pos, right_lane.line_base_pos)

    # 6) Output: warp lane boundaries back & display lane boundaries, curvature and position
    lanes_marked = alf.draw_lanes(top_view, undistorted, left_lane.average_fit, right_lane.average_fit, curvature,
                                  vehicle_position, Minv)

    # Set current values as previous values for next frame
    left_lane.previous_fit = left_lane.current_fit
    right_lane.previous_fit = right_lane.current_fit

    # Reset / empty current fit
    left_lane.current_fit = [np.array([False])]
    right_lane.current_fit = [np.array([False])]

    if mode is 'debug':
        debug_top = np.concatenate((img[:, 0:1279:2, :], lanes_marked[:, 0:1279:2, :]), axis=1)
        debug_bottom = np.concatenate((colored_thresholded[:, 0:1279:2, :], lanes_colored[:, 0:1279:2, :]), axis=1)
        debug = np.concatenate((debug_top[0:719:2], debug_bottom[0:719:2]), axis=0)
        return debug

    if mode is 'mark_lanes':
        return lanes_marked


# CAMERA CALIBRATION

# Set number of chessboard corners to find
nx = 9
ny = 6

print("calibrating camera...")

# Read in images
images = glob.glob('camera_cal/*.jpg')

# 0) Compute camera calibration matrix & distortion coefficients
cam_mtx, dist_coeffs = alf.camera_calibration(images, nx, ny)

# 1) Create line objects
left_lane = alf.Line()
right_lane = alf.Line()

# 2) Test pipeline
# TODO: set mode somewhere else?
# TODO: turn sanity check ON/OFF
# Set mode: mark_lanes OR debug
mode = 'debug'
# Test on image or video
test_pipeline('video2')