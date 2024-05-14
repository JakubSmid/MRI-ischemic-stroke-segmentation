"""
Script extracts the FLAIR images from the Motol dataset.
FLAIR images are stored together in one directory which is 
requested for bulk processing using HD-BET.
"""

import os
import shutil
import sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import datasets.dataset_loaders as dataset_loaders

motol = dataset_loaders.Motol()
os.makedirs("skull_stripping/Motol_FLAIR", exist_ok=True)

for subj in motol:
    shutil.copy(subj.flair, f"skull_stripping/Motol_FLAIR/{subj.name}.nii.gz")