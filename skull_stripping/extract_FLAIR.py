"""
Script extracts the FLAIR images from the Motol dataset.
FLAIR images are stored together in one directory which is 
requested for bulk processing using HD-BET.
"""

import os
import shutil
import sys

sys.path.append(os.getcwd())
import datasets.dataset_loaders as dataset_loaders

motol = dataset_loaders.Motol()

# create folder FLAIR if it does not exist
if not os.path.exists('skull_stripping/Motol_FLAIR'):
    os.mkdir('skull_stripping/Motol_FLAIR')

for flair, name in zip(motol.flairs, motol.names):
    shutil.copy(flair, f"skull_stripping/Motol_FLAIR/{name}.nii.gz")