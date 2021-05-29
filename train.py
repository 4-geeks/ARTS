import argparse
from config import createcsv
###########################
parser=argparse.ArgumentParser()

parser.add_argument('-in','--imagein',type=str,default='F:/praograming proje/file haye amozesh cv2/18_video/get_frame_from_video/sit_up')
parser.add_argument('-csv','--csvoutput',type=str,default='situp_csvs_out')
parser.add_argument('-out','--imageout',type=str,default='situp_poses_images_out')


args = parser.parse_args()

#########################
bootstrap_images_in_folder = args.imagein
bootstrap_csvs_out_folder = args.csvoutput
bootstrap_images_out_folder = args.imageout
csv=createcsv(bootstrap_images_in_folder,bootstrap_images_out_folder,bootstrap_csvs_out_folder)
csv.creat()

