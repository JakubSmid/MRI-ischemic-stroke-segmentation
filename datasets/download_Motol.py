"""
Script for downloading MRI scans and segmentation data 
for this project from NAS database.
"""

import shutil
import os

input_folder = "nas_data/"
output_folder = "usb_stick/"

patient_number_to_copy = [
    "98994",
    "2290116",
    "2173586",
    "145035",
    "2179097",
    "2260503",
    "2282310",
    "2290867"]

files_to_copy = [
    "rDWI1.nii",
    "rDWI2.nii",
    "rFlair.nii",
    "rqT1.nii",
    "rSWI.nii",
    "rT1.nii",
    "rT1KL.nii",
    "c1rT1.nii",
    "c2rT1.nii",
    "c3rT1.nii",
    "Leze_FLAIR_DWI2.nrrd"]
files_to_copy = [f.lower() for f in files_to_copy]

for folder in patient_number_to_copy:
    for root, dirs, files in os.walk(input_folder + folder):
        files = os.listdir(root)
        files = [f for f in files if os.path.isfile(root+'/'+f)]
        for f in files:
            if f.lower() in files_to_copy:
                dest = output_folder + root + "/" + f
                print(dest)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.copy(root+"/"+f, dest)
