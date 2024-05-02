import pandas as pd
import numpy as np
import nibabel as nib
import nibabel.processing

from torchmetrics.classification import BinaryStatScores, BinaryF1Score, MulticlassStatScores, MulticlassF1Score
import torch

import argparse
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from datasets.utils import voxel_count_to_volume_ml, subtract_masks, nrrd_to_nifti
import datasets.dataset_loaders

parser = argparse.ArgumentParser()
parser.add_argument("input_folder", type=str, help="Folder with predictions, each segmentation should have format {case}_Anat_{date}.nii.gz")
parser.add_argument("output_file", type=str, help="Output file name (.ods)")
args = parser.parse_args()

stats = MulticlassStatScores(num_classes=2, average="none", ignore_index=2)
dice = MulticlassF1Score(num_classes=2, average="none", ignore_index=2)

gt_dataset = datasets.dataset_loaders.Motol()
df = pd.DataFrame()
N = len(gt_dataset.names)
for i, (name, mask_file, bet_file) in enumerate(zip(gt_dataset.names, gt_dataset.masks, gt_dataset.BETmasks)):
    print(f"Processing {name} ({i+1}/{N})...")
    case = {}

    BETmask = nib.load(bet_file)
    mask_flair, mask_dwi = nrrd_to_nifti(mask_file)
    if mask_flair.shape != BETmask.shape:
        mask_flair = nibabel.processing.resample_from_to(mask_flair, BETmask, order=0)
        mask_dwi = nibabel.processing.resample_from_to(mask_dwi, BETmask, order=0)

    gt_label = np.logical_or(mask_flair.get_fdata(), mask_dwi.get_fdata())
    gt_label = nib.Nifti1Image(gt_label.astype(np.int8), affine=mask_flair.affine)
    gt_zooms = gt_label.header.get_zooms()

    pred_label = nib.load(f"{args.input_folder}/{name}.nii.gz")
    pred_label = nibabel.processing.resample_from_to(pred_label, gt_label, order=0)

    gt_label = gt_label.get_fdata().astype(np.uint8)
    gt_label[BETmask.get_fdata() == 0] = 2
    gt_label = torch.from_numpy(gt_label)

    gt_flair = mask_flair.get_fdata().astype(np.uint8)
    gt_flair[BETmask.get_fdata() == 0] = 2
    gt_flair = torch.from_numpy(gt_flair)

    gt_dwi = mask_dwi.get_fdata().astype(np.uint8)
    gt_dwi[BETmask.get_fdata() == 0] = 2
    gt_dwi = torch.from_numpy(gt_dwi)

    pred_label = pred_label.get_fdata().astype(np.uint8)
    pred_label[BETmask.get_fdata() == 0] = 2
    pred_label = torch.from_numpy(pred_label)

    tp, fp, tn, fn, support = voxel_count_to_volume_ml(stats(pred_label, gt_label).numpy()[1], gt_zooms)
    dc = dice(pred_label, gt_label).numpy()[1]
    case["tp"] = tp
    case["fp"] = fp
    case["tn"] = tn
    case["fn"] = fn
    case["dc"] = dc
    
    n_pred = (pred_label==1).sum().numpy()
    n_gt = (gt_label==1).sum().numpy()
    pred_volume = voxel_count_to_volume_ml(n_pred, gt_zooms)
    gt_volume = voxel_count_to_volume_ml(n_gt, gt_zooms)
    case["pred_volume"] = pred_volume
    case["gt_volume"] = gt_volume

    dc_flair = dice(pred_label, gt_flair).numpy()[1]
    dc_dwi = dice(pred_label, gt_dwi).numpy()[1]
    case["dc_flair"] = dc_flair
    case["dc_dwi"] = dc_dwi

    # FLAIR\(FLAIR + DWI) vs Y\(Y + DWI)
    gt_flair_only = subtract_masks(gt_flair, gt_dwi)
    pred_flair_only = subtract_masks(pred_label, gt_dwi)
    dc_flair = dice(pred_flair_only, gt_flair_only).numpy()[1]
    case["dc_flair_only"] = dc_flair

    # DWI\(DWI + FLAIR) vs Y\(Y + FLAIR)
    gt_dwi_only = subtract_masks(gt_dwi, gt_flair)
    pred_dwi_only = subtract_masks(pred_label, gt_flair)
    dc_dwi = dice(pred_dwi_only, gt_dwi_only).numpy()[1]
    case["dc_dwi_only"] = dc_dwi

    df = pd.concat([df, pd.DataFrame(case, index=[name])])

df.to_excel(args.output_file, engine="odf")
