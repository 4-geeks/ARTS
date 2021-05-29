import argparse

from matplotlib import pyplot as plt
import cv2
from config import Observer

#######################
parser=argparse.ArgumentParser()

parser.add_argument('-video','--videoin',type=str,default='ww.mp4')
parser.add_argument('-csv','--csvin',type=str,default='situp_csvs_out')
parser.add_argument('-dic','--dicaction',type=str,default={'act1':['sit','up']})


args = parser.parse_args()
###################

video_path = args.videoin
pose_samples_folder = args.csvin
video_cap = cv2.VideoCapture(video_path)
fps = video_cap.get(cv2.CAP_PROP_FPS)
DIC_ACT=args.dicaction
obsrvr=Observer(pose_samples_folder,fps,DIC_ACT)
nm=obsrvr.name

while True:
  success, input_frame = video_cap.read()
  if not success:
    break
  obsrvr.update(input_frame,second=1,th_score=6)
  obsrvr.inference()
  obsrvr.SaveVideo(save=True)