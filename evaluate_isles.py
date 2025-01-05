import pandas as pd
import numpy as np
import ants
import argparse

from torchmetrics.classification import MulticlassStatScores, MulticlassF1Score
import torch

import datasets.utils as utils
import datasets.dataset_loaders as dataset_loaders

def load_data(subject):
    flair = ants.image_read(subject.flair)
    transform = ants.read_transform(subject.transform_dwi_to_flair)

    label = ants.image_read(subject.label).astype("uint32")
    if subject.labeled_modality == "dwi":
        label = utils.apply_transform_to_label(label, transform, flair)

    BETmask = flair.new_image_like((flair.numpy() != 0).astype("uint32"))
    return label, BETmask

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder", type=str, help="Folder with predictions, each segmentation should have format {case}_Anat_{date}.nii.gz")
    parser.add_argument("output_file", type=str, help="Output file name (csv)")
    parser.add_argument("--mni", action="store_true", help="Predictions are in MNI space")
    args = parser.parse_args()

    # ignore index 2 (outside BET mask) - it is important for corect Stat scores (true positives in ml, etc.)
    stats = MulticlassStatScores(num_classes=2, average="none", ignore_index=2)
    dice = MulticlassF1Score(num_classes=2, average="none", ignore_index=2)

    gt_dataset = dataset_loaders.ISLES2015() + dataset_loaders.ISLES2022()
    df = pd.DataFrame()
    N = len(gt_dataset)
    for i, subj in enumerate(gt_dataset):
        print(f"Processing {subj.name} ({i+1}/{N})...")
        case = {}

        # load ground truth labels
        label, BETmask = load_data(subj)

        # load prediction
        pred_label = ants.image_read(f"{args.input_folder}/{subj.name}.nii.gz")
        
        # transform from MNI space
        if args.mni:
            pred_label = utils.invert_SyN_registration(pred_label.astype("float32"), subj.transform_flair_to_mni[0], subj.transform_flair_to_mni[1])
            pred_label = pred_label.new_image_like(pred_label.numpy().round().astype(np.uint32))

        pred_label = utils.resample_label_to_target(pred_label, label.astype("float32"))
        assert label.shape == pred_label.shape, f"Shape mismatch: {label.shape} != {pred_label.shape}"
        assert label.spacing == pred_label.spacing, f"Spacing mismatch: {label.spacing} != {pred_label.spacing}"

        # transfrorm to tensor
        gt_label = label.numpy().astype(np.uint8)
        gt_label[BETmask.numpy() == 0] = 2
        gt_label = torch.from_numpy(gt_label)

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
        
        df = pd.concat([df, pd.DataFrame(case, index=[subj.name])])

    df.to_csv(args.output_file)
