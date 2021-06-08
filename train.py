from tool import createcsv
from config import inputimagefolder,csvoutputfolder,outpoutimagefolder

#########################
bootstrap_images_in_folder = inputimagefolder
bootstrap_csvs_out_folder = csvoutputfolder
bootstrap_images_out_folder = outpoutimagefolder
csv=createcsv(bootstrap_images_in_folder,bootstrap_images_out_folder,bootstrap_csvs_out_folder)
csv.creat()

