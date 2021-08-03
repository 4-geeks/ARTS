from matplotlib import pyplot as plt
import cv2
from tool import Observer
video_path = 0 #'/content/drive/MyDrive/data/test/ff.mp4'
pose_samples_folder = 'pos_csvs'
###
import ffmpeg    

def check_rotation(path_video_file):
    # this returns meta-data of the video file in form of a dictionary
    meta_dict = ffmpeg.probe(path_video_file)

    # from the dictionary, meta_dict['streams'][0]['tags']['rotate'] is the key
    # we are looking for
    rotateCode = None
    try:
      if int(meta_dict['streams'][0]['tags']['rotate']) == 90:
          rotateCode = cv2.ROTATE_90_CLOCKWISE
      elif int(meta_dict['streams'][0]['tags']['rotate']) == 180:
          rotateCode = cv2.ROTATE_180
      elif int(meta_dict['streams'][0]['tags']['rotate']) == 270:
          rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE
    except:
       pass

    return rotateCode
def correct_rotation(frame, rotateCode):  
    return cv2.rotate(frame, rotateCode) 
###
video_cap = cv2.VideoCapture(video_path)
# rotateCode = check_rotation(video_path)
fps = video_cap.get(cv2.CAP_PROP_FPS)
DIC_ACT={'zoomout':['zoom_close','hand_top'],'zoomin':['hand_top','zoom_close'],
         'swipeleft':['hand_left','hand_right_swipe'],'swiperight':['hand_right_swipe','hand_left'],
         'scrollup':['hand_down','hand_left','hand_top'],'scrolldown':['hand_down','hand_left','hand_down']
         }
name_out_video='ff_test.avi'
###
obsrvr=Observer(pose_samples_folder,fps,DIC_ACT,name_out_video)
nm=obsrvr.name
while True:
  success, input_frame = video_cap.read()
  if not success:
    break
#   if rotateCode is not None:
#          input_frame = correct_rotation(input_frame, rotateCode)

  obsrvr.update(input_frame,second=4.5,th_score=8,frame_stop_second=1.5)

  obsrvr.inference()
  im = obsrvr.SaveVideo(save=True)
  cv2.imshow('s',im)
  if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cv2.destroyAllWindows()