"""
Script copies generated BET masks back to the Motol dataset.
For each patient, the BET mask is copied to the corresponding folder.
"""

import os
import sys
import shutil
import gzip

for file in os.listdir('skull_stripping/Motol_FLAIR'):
    # skip _bet folder with output of HD-BET
    if file == "_bet":
        continue

    # copy FLAIR BET mask
    if "_mask" in file:
        # get patient folder name and anat
        patient_folder = file.split("_")[0]
        anat = file[len(patient_folder) + 1:-12]

        src = f"skull_stripping/Motol_FLAIR/{file}"
        dst = f"datasets/Motol/{patient_folder}/{anat}/BETmask.nii.gz"
        print(f"Copying {src} -> {dst}")
        shutil.copyfile(src, dst)
