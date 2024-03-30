import pandas as pd
import nibabel as nib
import numpy as np

import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from datasets.utils import voxel_count_to_volume_ml, dice_coefficient, x_without_y

def evaluate_model(model_folder):
    """
    Evaluates a model by calculating various metrics for each case in the model folder.
    
    Args:
        model_folder (str): The path to the folder containing the trained model and predictions.
        
    Returns:
        pandas.DataFrame: A DataFrame containing the calculated metrics for each case.
    """
    # create list with paths to segmentations
    cases = [case.replace("_Segm", "") for case in os.listdir(model_folder + "/predictions") if case.endswith("Segm.nii.gz")]
    df = pd.DataFrame()
    
    for case in cases:
        y = nib.load(f"{model_folder}/predictions/{case.split('.')[0]}_Segm.nii.gz")

        # load ground truth masks
        gt_file = case.replace("_", "/", 1).split(".")[0]
        gt = nib.load(f"deepmedic_workspace/preprocessed/Motol/{gt_file}/mask.nii.gz")
        gt_dwi = nib.load(f"deepmedic_workspace/preprocessed/Motol/{gt_file}/mask_dwi.nii.gz")
        gt_flair = nib.load(f"deepmedic_workspace/preprocessed/Motol/{gt_file}/mask_flair.nii.gz")

        # calculate volumes
        case_dict = {
            "pred_volume": voxel_count_to_volume_ml(np.count_nonzero(y.get_fdata()), y.header.get_zooms()),
            "gt_volume": voxel_count_to_volume_ml(np.count_nonzero(gt.get_fdata()), gt.header.get_zooms()),
        }

        # from SpatialImage to numpy 
        y = y.get_fdata()
        gt = gt.get_fdata()
        gt_dwi = gt_dwi.get_fdata()
        gt_flair = gt_flair.get_fdata()

        # calculate Dice coefficients
        case_dict['dice'] = dice_coefficient(gt, y)
        case_dict['dice_flair'] = dice_coefficient(gt_flair, y)
        case_dict['dice_dwi'] = dice_coefficient(gt_dwi, y)
        
        # DWI\(DWI + FLAIR) vs Y\(Y + FLAIR)
        dwi_only = x_without_y(gt_dwi, gt_flair)
        y_dwi_only = x_without_y(y, gt_flair)
        case_dict['dice_dwi_only'] = dice_coefficient(dwi_only, y_dwi_only)

        # FLAIR\(FLAIR + DWI) vs Y\(Y + DWI)
        flair_only = x_without_y(gt_flair, gt_dwi)
        y_flair_only = x_without_y(y, gt_dwi)
        case_dict['dice_flair_only'] = dice_coefficient(flair_only, y_flair_only)

        # add current case to the DataFrame
        df = pd.concat([df, pd.DataFrame(case_dict, index=[case])])
    return df

if __name__ == "__main__":
    # folder with predictions for specific model
    models = "deepmedic_workspace/output/predictions/"
    
    with pd.ExcelWriter("deepmedic_results.ods", engine="odf") as writer:
        for model_folder in os.listdir(models):
            df = evaluate_model(models + model_folder)
            df.to_excel(writer, sheet_name=model_folder)
