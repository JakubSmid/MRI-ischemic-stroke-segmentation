"""
Script for downloading MRI scans and segmentation data 
for this project from NAS database.
"""

import shutil
import os

input_folder = "/run/user/1000/gvfs/smb-share:server=195.113.42.186,share=data3/motol3Texp/BEAST3/"
output_folder = "Motol/"

files_to_copy = [
    "rDWI2.nii",
    "rFlair.nii",
    "Leze_FLAIR_DWI2.nrrd",
    "Leze_FLAIR_DWI.nrrd"]
files_to_copy = [f.lower() for f in files_to_copy]

for folder in os.listdir(input_folder):
    if not folder.isnumeric():
        continue
    for root, dirs, files in os.walk(input_folder + folder):
        files = os.listdir(root)
        files = [f for f in files if os.path.isfile(root+'/'+f)]
        for f in files:
            if f.lower() in files_to_copy:
                dest = output_folder + root + "/" + f
                print(dest)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.copy(root+"/"+f, dest)
