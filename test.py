import P2_Pipeline
import AdvancedLaneFinding as alf

# 1) Create line / lane objects
# TODO: put this into test_pipeline/image_pipeline/video_pipeline
left_lane = alf.Line()
right_lane = alf.Line()
lane = alf.Lane()

# 2) Test pipeline
# TODO: set mode somewhere else?
# TODO: turn sanity check ON/OFF
# Set mode: mark_lanes OR debug
mode = 'debug'
# Test on image or video
P2_Pipeline.test_pipeline('video2')