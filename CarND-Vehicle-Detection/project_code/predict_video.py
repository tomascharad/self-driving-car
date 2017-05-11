from multiple_detections import *
from moviepy.editor import VideoFileClip
from collections import deque
from predict import get_bboxes
import numpy as np

def process_without_state():
  clip1 = VideoFileClip("../test_video.mp4")
  white_clip = clip1.fl_image(sinlge_image_detection) #NOTE: this function expects color images!!
  white_clip.write_videofile('../test_output/video.mp4', audio=False)

def process_with_state():
  clip1 = VideoFileClip("../project_video.mp4")
  white_clip = clip1.fl_image(detection_with_state) #NOTE: this function expects color images!!
  white_clip.write_videofile('../test_output/video_state_final.mp4', audio=False)

d = deque(maxlen = 10)
def detection_with_state(img, vis=False):
  find_car_bboxes = get_bboxes(img)
  heat = np.zeros_like(img[:,:,0]).astype(np.float)
  heatmap = add_heat(heat, find_car_bboxes)
  d.append(heatmap)
  heatmap = np.average(d, axis=0)
  heatmap = apply_threshold(heatmap, 3)
  cliped_heatmap = np.clip(heatmap, 0, 255)
  labels = label(cliped_heatmap)
  draw_img = draw_labeled_bboxes(np.copy(img), labels)
  to_return = draw_img
  if vis:
    to_return = [draw_img, cliped_heatmap]
  return to_return

print('Starting')
process_with_state()
print('End')