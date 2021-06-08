from matplotlib import pyplot as plt
import cv2
from tool import Observer
from config import video,csvinputfolder,actiondict,th_score,second,save_video


video_path = video
pose_samples_folder = csvinputfolder
video_cap = cv2.VideoCapture(video_path)
fps = video_cap.get(cv2.CAP_PROP_FPS)
DIC_ACT=actiondict
obsrvr=Observer(pose_samples_folder,fps,DIC_ACT)
nm=obsrvr.name

while True:
  success, input_frame = video_cap.read()
  if not success:
    break
  obsrvr.update(input_frame,second=second,th_score=th_score)
  obsrvr.inference()
  obsrvr.SaveVideo(save=save_video)