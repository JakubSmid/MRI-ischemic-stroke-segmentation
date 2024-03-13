import nibabel as nib
import nibabel.processing
import ants
import cc3d
import os
import numpy as np
import pandas as pd
from utils import *
import dataset_loaders
import multiprocessing

def deepmedic_preprocess(flair_images: tuple[str, ...],
                         dwi_images: tuple[str, ...],
                         masks: tuple[str, ...],
                         fixed: str,
                         case_names: tuple[str, ...],
                         dataset_name: str,
                         BETmasks: tuple[str]|None = None,
                         output_folder: str = "preprocessed/"):
    """
    Preprocesses the input MRI images and masks, computes statistics, performs brain extraction, reshapes images, normalizes scans, calculates dice coefficient, and saves the preprocessed images and statistics to the output folder.

    Args:
    - flair_images (tuple[str, ...]): Paths to the FLAIR images.
    - dwi_images (tuple[str, ...]): Paths to the DWI images.
    - masks (tuple[str, ...]): Paths to the masks.
    - fixed (str): Indicates whether to use 'flair' or 'dwi' as the fixed image.
    - case_names (tuple[str, ...]): Names of the cases.
    - dataset_name (str): Name of the dataset.
    - BETmasks (tuple[str]|None, optional): Paths to the brain extraction masks. If BET masks are provided, images have to be co-registered! Defaults to None.
    - output_folder (str, optional): Output folder path. Defaults to "preprocessed/".

    Returns:
    - None
    """
    assert fixed in ["flair", "dwi"], "fixed must be 'flair' or 'dwi'"
    statistics = {
        "name": [],
        "shape_flair": [],
        "shape_dwi": [],
        "voxel_dim_flair": [],
        "voxel_dim_dwi": [],
        "volume_ml": [],
        "number_of_components": [],
        "dice_after_preprocessing": [],
    }
    N = len(flair_images)
    i = 0

    for flair_image, dwi_image, fixed_mask, name in zip(flair_images, dwi_images, masks, case_names):
        # print progress
        i += 1
        print(f"{dataset_name}: Processing {name} ({i}/{N})...")

        # load images
        flair = nib.load(flair_image)
        dwi = nib.load(dwi_image)

        # load mask
        if ".nrrd" in fixed_mask:
            mask_flair, mask_dwi = nrrd_to_nifti(fixed_mask, flair.affine)
            mask = np.logical_or(mask_flair.get_fdata(), mask_dwi.get_fdata())
            mask = nib.Nifti1Image(mask.astype(np.int8), affine=flair.affine)
        else:
            mask = nib.load(fixed_mask)
        orig_mask = mask

        # compute statistics
        components = cc3d.connected_components(mask.get_fdata(), connectivity=26)
        if fixed == "flair":
            volume = voxel_count_to_volume_ml(np.count_nonzero(mask.get_fdata()), flair.header.get_zooms())
        elif fixed == "dwi":
            volume = voxel_count_to_volume_ml(np.count_nonzero(mask.get_fdata()), dwi.header.get_zooms())

        # add statistics to dictionary
        statistics["name"].append(f"{name}")
        statistics["shape_flair"].append(flair.shape)
        statistics["shape_dwi"].append(dwi.shape)
        statistics["voxel_dim_flair"].append(flair.header.get_zooms())
        statistics["voxel_dim_dwi"].append(dwi.header.get_zooms())
        statistics["volume_ml"].append(volume)
        statistics["number_of_components"].append(components.max())

        # brain extraction: use HD-BET output or filter values greater than 0
        # If BET masks are provided, images have to be co-registered!
        if BETmasks is not None:
            ROImask = nib.load(BETmasks[i-1])
            # DeepMedic bug workaround: recasting ROI from uint8 to int8
            ROImask = nib.Nifti1Image(ROImask.get_fdata().astype(np.int8), affine=ROImask.affine)
            flair = nib.Nifti1Image(flair.get_fdata() * ROImask.get_fdata(), affine=flair.affine)
            dwi = nib.Nifti1Image(dwi.get_fdata() * ROImask.get_fdata(), affine=dwi.affine)
            mask = nib.Nifti1Image((mask.get_fdata() * ROImask.get_fdata()).astype(np.int8), affine=mask.affine)
            if ".nrrd" in fixed_mask:
                mask_flair = nib.Nifti1Image((mask_flair.get_fdata() * ROImask.get_fdata()).astype(np.int8), affine=mask_flair.affine)
                mask_dwi = nib.Nifti1Image((mask_dwi.get_fdata() * ROImask.get_fdata()).astype(np.int8), affine=mask_dwi.affine)
        elif fixed == "flair":
            ROImask = nib.Nifti1Image((flair.get_fdata() > 0).astype(np.int8), affine=flair.affine)
        elif fixed == "dwi":
            ROImask = nib.Nifti1Image((dwi.get_fdata() > 0).astype(np.int8), affine=dwi.affine)

        # reshape masks
        ROImask = nibabel.processing.conform(ROImask, voxel_size=(1,1,1), out_shape=(200,200,200), order=0)
        mask = nibabel.processing.conform(mask, voxel_size=(1,1,1), out_shape=(200,200,200), order=0)
        if ".nrrd" in fixed_mask:
            mask_flair = nibabel.processing.conform(mask_flair, voxel_size=(1,1,1), out_shape=(200,200,200), order=0)
            mask_dwi = nibabel.processing.conform(mask_dwi, voxel_size=(1,1,1), out_shape=(200,200,200), order=0)

        # perform registration
        if fixed == "flair":
            # reshape fixed image
            flair = nibabel.processing.conform(flair, voxel_size=(1,1,1), out_shape=(200,200,200))
            # from nibabel to ants
            flair = ants.from_nibabel(flair)
            dwi = ants.from_nibabel(dwi)
            # registration
            transform = ants.registration(fixed=flair, moving=dwi, type_of_transform = 'Rigid')
            dwi = ants.apply_transforms(fixed=flair, moving=dwi, transformlist=transform['fwdtransforms'])
        elif fixed == "dwi":
            # reshape fixed image
            dwi = nibabel.processing.conform(dwi, voxel_size=(1,1,1), out_shape=(200,200,200))
            # from nibabel to ants
            flair = ants.from_nibabel(flair)
            dwi = ants.from_nibabel(dwi)
            # registration
            transform = ants.registration(fixed=dwi, moving=flair, type_of_transform = 'Rigid')
            flair = ants.apply_transforms(fixed=dwi, moving=flair, transformlist=transform['fwdtransforms'])

        # from ants back to nibabel
        flair = ants.to_nibabel(flair)
        dwi = ants.to_nibabel(dwi)

        # normalize scans within ROI
        n_flair = flair.get_fdata()[ROImask.get_fdata() == 1]
        n_flair = (flair.get_fdata() - n_flair.mean()) / n_flair.std()
        n_dwi = dwi.get_fdata()[ROImask.get_fdata() == 1]
        n_dwi = (dwi.get_fdata() - n_dwi.mean()) / n_dwi.std()
        flair = nib.Nifti1Image(n_flair, affine=flair.affine)
        dwi = nib.Nifti1Image(n_dwi, affine=dwi.affine)

        # check dimensions, mean and std
        assert flair.shape == dwi.shape == mask.shape, f"Shapes are not the same: {flair.shape}, {dwi.shape}, {mask.shape}"
        assert np.allclose(flair.get_fdata()[ROImask.get_fdata() == 1].mean(), 0) and np.allclose(flair.get_fdata()[ROImask.get_fdata() == 1].std(), 1), \
            f"Mean and std within ROI of flair image are not 0 and 1: {flair.get_fdata()[ROImask.get_fdata() == 1].mean()}, {flair.get_fdata()[ROImask.get_fdata() == 1].std()}"
        assert np.allclose(dwi.get_fdata()[ROImask.get_fdata() == 1].mean(), 0) and np.allclose(dwi.get_fdata()[ROImask.get_fdata() == 1].std(), 1), \
            f"Mean and std within ROI of dwi image are not 0 and 1: {dwi.get_fdata()[ROImask.get_fdata() == 1].mean()}, {dwi.get_fdata()[ROImask.get_fdata() == 1].std()}"

        # check error after preprocessing
        mask_interpol = nibabel.processing.resample_from_to(mask, orig_mask, order=0)
        statistics["dice_after_preprocessing"].append(dice_coefficient(orig_mask, mask_interpol))

        # prepare folder
        if not os.path.exists(f"{output_folder}/{dataset_name}/{name}"):
            os.makedirs(f"{output_folder}/{dataset_name}/{name}")

        # save images
        nib.save(mask, f"{output_folder}/{dataset_name}/{name}/mask.nii.gz")
        nib.save(flair, f"{output_folder}/{dataset_name}/{name}/flair.nii.gz")
        nib.save(dwi, f"{output_folder}/{dataset_name}/{name}/dwi.nii.gz")
        nib.save(ROImask, f"{output_folder}/{dataset_name}/{name}/ROImask.nii.gz")
        if ".nrrd" in fixed_mask:
            nib.save(mask_flair, f"{output_folder}/{dataset_name}/{name}/mask_flair.nii.gz")
            nib.save(mask_dwi, f"{output_folder}/{dataset_name}/{name}/mask_dwi.nii.gz")

    # save statistics
    pd.DataFrame(statistics).to_excel(f"{output_folder}/{dataset_name}_statistics.ods", index=False)

if __name__ == "__main__":
    # load datasets
    motol = dataset_loaders.Motol()
    isles2015 = dataset_loaders.ISLES2015()
    isles2022 = dataset_loaders.ISLES2022()

    # run preprocessing for each dataset in parallel
    p1 = multiprocessing.Process(target=deepmedic_preprocess, 
                                 args=(motol.flairs, motol.dwis, motol.masks, "flair", motol.names, "Motol", motol.BETmasks))

    p2 = multiprocessing.Process(target=deepmedic_preprocess,
                                 args=(isles2015.flairs, isles2015.dwis, isles2015.masks, "flair", isles2015.names, "ISLES2015"))

    p3 = multiprocessing.Process(target=deepmedic_preprocess,
                                 args=(isles2022.flairs, isles2022.dwis, isles2022.masks, "dwi", isles2022.names, "ISLES2022"))
    
    p1.start()
    p2.start()
    p3.start()

    p1.join()
    p2.join()
    p3.join()