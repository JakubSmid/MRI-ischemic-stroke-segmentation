import os
import json
import logging
import nibabel as nib
import numpy as np

import torchio as tio

logger = logging.getLogger(__name__)

def load_dataset(dataset_dir="./raw", fold=0, mode="train", transform=None, exclude_empty=False):
    splits = json.load(open("./splits.json"))
    split = splits[fold][mode]

    subjects = []
    for name in split:
        if exclude_empty and name in ["sub-strokecase0150", "sub-strokecase0151", "sub-strokecase0170"]:
            continue

        subj_dict = {"name": name}
        # load images
        subj_dict["flair"] = tio.ScalarImage(os.path.join(dataset_dir, f"{name}/flair.nii.gz"))
        subj_dict["dwi"] = tio.ScalarImage(os.path.join(dataset_dir, f"{name}/dwi.nii.gz"))
        
        # load BET mask and label
        subj_dict["label"] = tio.LabelMap(os.path.join(dataset_dir, f"{name}/label.nii.gz"))
        subj_dict["BETmask"] = tio.LabelMap(os.path.join(dataset_dir, f"{name}/BETmask.nii.gz"))

        subject = tio.Subject(subj_dict)
        subjects.append(subject)

    return tio.SubjectsDataset(subjects, transform=transform)
