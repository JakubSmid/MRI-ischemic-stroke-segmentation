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

    # get patient folder name and anat
    patient_folder = file.split("_")[0]
    anat = file[len(patient_folder) + 1:].split(".")[0]

    # copy FLAIR BET mask
    with gzip.open(f"skull_stripping/Motol_FLAIR/_bet/{patient_folder}_{anat}_mask.nii.gz", "rb") as f_in:
        with open(f"datasets/Motol/{patient_folder}/{anat}/BETmask.nii", "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
