import pandas as pd
import numpy as np
import ants
import argparse

from torchmetrics.classification import MulticlassStatScores, MulticlassF1Score
import torch

import datasets.utils as utils
import datasets.dataset_loaders as dataset_loaders

def load_label(subject: dataset_loaders.Subject):
    transform = ants.read_transform(subject.transform_dwi_to_flair)
    BETmask = ants.image_read(subject.BETmask).astype("uint32")

    label_flair, label_dwi = utils.load_nrrd(subject.label)

    # resample flair label to BETmask
    if label_flair.shape != BETmask.shape:
        label_flair = utils.resample_label_to_target(label_flair, BETmask.astype("float32"))

    # resample dwi label to flair
    if label_dwi.shape != BETmask.shape:
        label_dwi = utils.resample_label_to_target(label_dwi, BETmask.astype("float32"))
    
    label_dwi = utils.apply_transform_to_label(label_dwi, transform, BETmask.astype("float32"))

    label_union = np.logical_or(label_flair.numpy(), label_dwi.numpy()).astype(np.uint32)
    label = label_flair.new_image_like(label_union)
    return label, label_dwi, label_flair, BETmask

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder", type=str, help="Folder with predictions, each segmentation should have format {case}_Anat_{date}.nii.gz")
    parser.add_argument("output_file", type=str, help="Output file name (csv)")
    args = parser.parse_args()

    # ignore index 2 (outside BET mask) - it is important for corect Stat scores (true positives in ml, etc.)
    stats = MulticlassStatScores(num_classes=2, average="none", ignore_index=2)
    dice = MulticlassF1Score(num_classes=2, average="none", ignore_index=2)

    gt_dataset = dataset_loaders.Motol()
    df = pd.DataFrame()
    N = len(gt_dataset)
    for i, subj in enumerate(gt_dataset):
        print(f"Processing {subj.name} ({i+1}/{N})...")
        case = {}

        # load ground truth labels
        label, gt_dwi, gt_flair, BETmask = load_label(subj)

        # load prediction
        pred_label = ants.image_read(f"{args.input_folder}/{subj.name}.nii.gz")
        pred_label = utils.resample_label_to_target(pred_label, label.astype("float32"))
        assert label.shape == pred_label.shape, f"Shape mismatch: {label.shape} != {pred_label.shape}"
        assert label.spacing == pred_label.spacing, f"Spacing mismatch: {label.spacing} != {pred_label.spacing}"

        # transfrorm to tensor
        gt_label = label.numpy().astype(np.uint8)
        gt_label[BETmask.numpy() == 0] = 2
        gt_label = torch.from_numpy(gt_label)

        gt_flair = gt_flair.numpy().astype(np.uint8)
        gt_flair[BETmask.numpy() == 0] = 2
        gt_flair = torch.from_numpy(gt_flair)

        gt_dwi = gt_dwi.numpy().astype(np.uint8)
        gt_dwi[BETmask.numpy() == 0] = 2
        gt_dwi = torch.from_numpy(gt_dwi)

        pred_label = pred_label.numpy().astype(np.uint8)
        pred_label[BETmask.numpy() == 0] = 2
        pred_label = torch.from_numpy(pred_label)

        tp, fp, tn, fn, support = utils.voxel_count_to_volume_ml(stats(pred_label, gt_label).numpy()[1], label.spacing)
        dc = dice(pred_label, gt_label).numpy()[1]
        case["tp"] = tp
        case["fp"] = fp
        case["tn"] = tn
        case["fn"] = fn
        case["dc"] = dc
        
        n_pred = (pred_label==1).sum().numpy()
        n_gt = (gt_label==1).sum().numpy()
        pred_volume = utils.voxel_count_to_volume_ml(n_pred, label.spacing)
        gt_volume = utils.voxel_count_to_volume_ml(n_gt, label.spacing)
        case["pred_volume"] = pred_volume
        case["gt_volume"] = gt_volume

        dc_flair = dice(pred_label, gt_flair).numpy()[1]
        dc_dwi = dice(pred_label, gt_dwi).numpy()[1]
        case["dc_flair"] = dc_flair
        case["dc_dwi"] = dc_dwi

        df = pd.concat([df, pd.DataFrame(case, index=[subj.name])])

    df.to_csv(args.output_file)
