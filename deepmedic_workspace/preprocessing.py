import nibabel as nib
import nibabel.processing
import ants
import cc3d
import numpy as np
import pandas as pd
import multiprocessing

import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from datasets.utils import *
import datasets.dataset_loaders as dataset_loaders

def deepmedic_preprocess_Motol(flair_images: tuple[str, ...],
                               dwi_images: tuple[str, ...],
                               masks: tuple[str, ...],
                               case_names: tuple[str, ...],
                               BETmasks: tuple[str, ...],
                               dataset_name: str = "Motol",
                               output_folder: str = "deepmedic_workspace/preprocessed/"):
    """
    Preprocesses the Motol dataset for the DeepMedic project.

    Args:
        flair_images (tuple[str, ...]): A tuple of paths to FLAIR images.
        dwi_images (tuple[str, ...]): A tuple of paths to DWI images.
        masks (tuple[str, ...]): A tuple of paths to masks.
        case_names (tuple[str, ...]): A tuple of case names.
        BETmasks (tuple[str]): A tuple of paths to BET masks.
        dataset_name (str, optional): The name of the dataset. Defaults to "Motol".
        output_folder (str, optional): The output folder path. Defaults to "deepmedic_workspace/preprocessed/".
        
    Returns:
        None
    """
    df_cases = pd.DataFrame()
    df_components = pd.DataFrame()

    N = len(flair_images)
    for i, (flair_image, dwi_image, fixed_mask, BET_mask, name) in enumerate(zip(flair_images, dwi_images, masks, BETmasks, case_names)):
        # print progress
        print(f"{dataset_name}: Processing {name} ({i+1}/{N})...")

        # load images and masks
        flair = nib.load(flair_image)
        dwi = nib.load(dwi_image)
        mask_flair, mask_dwi = nrrd_to_nifti(fixed_mask)
        mask = np.logical_or(mask_flair.get_fdata(), mask_dwi.get_fdata())
        orig_mask = nib.Nifti1Image(mask.astype(np.int8), affine=mask_flair.affine)
        mask = nib.Nifti1Image(mask.astype(np.int8), affine=mask_flair.affine)
        
        # compute components
        components = cc3d.connected_components(mask.get_fdata(), connectivity=26)
        for component in np.unique(components)[1:]:
            volume = voxel_count_to_volume_ml(np.count_nonzero(components == component), flair.header.get_zooms())
            stats = {
                "name": name,
                "volume_ml": volume
            }
            df_components = pd.concat([df_components, pd.DataFrame([stats])])

        # compute volumes
        volume = voxel_count_to_volume_ml(np.count_nonzero(mask.get_fdata()), flair.header.get_zooms())

        # save metadata to dict
        stats = {
            "name": name,
            "shape_flair": flair.shape,
            "shape_dwi": dwi.shape,
            "voxel_dim_flair": flair.header.get_zooms(),
            "voxel_dim_dwi": dwi.header.get_zooms(),
            "lesion_volume_ml": volume,
            "flair_lesion_volume_ml": voxel_count_to_volume_ml(np.count_nonzero(mask_flair.get_fdata()), flair.header.get_zooms()),
            "dwi_lesion_volume_ml": voxel_count_to_volume_ml(np.count_nonzero(mask_dwi.get_fdata()), flair.header.get_zooms()),
        }

        # reshape flair
        flair = nibabel.processing.conform(flair, voxel_size=(1,1,1), out_shape=(200,200,200))

        # from nibabel to ants
        flair = ants.from_nibabel(flair)
        dwi = ants.from_nibabel(dwi)

        # registration
        transform = ants.registration(fixed=flair, moving=dwi, type_of_transform = 'Rigid')
        dwi = ants.apply_transforms(fixed=flair, moving=dwi, transformlist=transform['fwdtransforms'])

        # from ants back to nibabel
        flair = ants.to_nibabel(flair)
        dwi = ants.to_nibabel(dwi)

        # load BET mask
        # assume BET masks are co-registered to FLAIR!
        ROImask = nib.load(BET_mask)
        # DeepMedic bug -> recast from uint to int
        ROImask = nib.Nifti1Image(ROImask.get_fdata().astype(np.int8), affine=ROImask.affine)
        
        # reshape masks
        ROImask = nibabel.processing.resample_from_to(ROImask, flair, order=0)
        mask = nibabel.processing.resample_from_to(mask, flair, order=0)

        # apply BET mask
        flair = nib.Nifti1Image(flair.get_fdata() * ROImask.get_fdata(), affine=flair.affine)
        dwi = nib.Nifti1Image(dwi.get_fdata() * ROImask.get_fdata(), affine=dwi.affine)
        mask = nib.Nifti1Image((mask.get_fdata() * ROImask.get_fdata()).astype(np.int8), affine=mask.affine)

        stats["bet_mask_volume_ml"] = voxel_count_to_volume_ml(np.count_nonzero(ROImask.get_fdata()), ROImask.header.get_zooms())

        # normalize scans within ROI
        n_flair = flair.get_fdata()[ROImask.get_fdata() == 1]
        n_flair = (flair.get_fdata() - n_flair.mean()) / n_flair.std()
        n_dwi = dwi.get_fdata()[ROImask.get_fdata() == 1]
        n_dwi = (dwi.get_fdata() - n_dwi.mean()) / n_dwi.std()
        flair = nib.Nifti1Image(n_flair, affine=flair.affine)
        dwi = nib.Nifti1Image(n_dwi, affine=dwi.affine)

        # check dimensions, mean and std
        assert np.allclose(flair.get_fdata()[ROImask.get_fdata() == 1].mean(), 0) and np.allclose(flair.get_fdata()[ROImask.get_fdata() == 1].std(), 1), \
            f"Mean and std within ROI of flair image are not 0 and 1: {flair.get_fdata()[ROImask.get_fdata() == 1].mean()}, {flair.get_fdata()[ROImask.get_fdata() == 1].std()}"
        assert np.allclose(dwi.get_fdata()[ROImask.get_fdata() == 1].mean(), 0) and np.allclose(dwi.get_fdata()[ROImask.get_fdata() == 1].std(), 1), \
            f"Mean and std within ROI of dwi image are not 0 and 1: {dwi.get_fdata()[ROImask.get_fdata() == 1].mean()}, {dwi.get_fdata()[ROImask.get_fdata() == 1].std()}"
        assert flair.shape == dwi.shape == mask.shape, f"Shapes are not the same: {flair.shape}, {dwi.shape}, {mask.shape}"
        assert np.allclose(flair.affine, dwi.affine), f"FLAIR and DWI have different affines:\n{flair.affine}\n{dwi.affine}"
        assert np.allclose(flair.affine, mask.affine), f"FLAIR and mask have different affines:\n{flair.affine}\n{mask.affine}"
        assert (mask.get_fdata()==1).any(), "Mask is empty"

        # check error after preprocessing
        mask_interpol = nibabel.processing.resample_from_to(mask, orig_mask, order=0)
        stats["dice_after_preprocessing"] = dice_coefficient(orig_mask, mask_interpol)
        df_cases = pd.concat([df_cases, pd.DataFrame([stats])])

        # prepare folder
        if not os.path.exists(f"{output_folder}/{dataset_name}/{name}"):
            os.makedirs(f"{output_folder}/{dataset_name}/{name}")

        # save images
        nib.save(mask, f"{output_folder}/{dataset_name}/{name}/label.nii.gz")
        nib.save(flair, f"{output_folder}/{dataset_name}/{name}/flair.nii.gz")
        nib.save(dwi, f"{output_folder}/{dataset_name}/{name}/dwi.nii.gz")
        nib.save(ROImask, f"{output_folder}/{dataset_name}/{name}/ROImask.nii.gz")

    # save dataframes
    with pd.ExcelWriter(f"{output_folder}/{dataset_name}_metadata.ods") as writer:
        df_cases.to_excel(writer, sheet_name="cases", index=False)
        df_components.to_excel(writer, sheet_name="components", index=False)

def deepmedic_preprocess_ISLES(flair_images: tuple[str, ...],
                               dwi_images: tuple[str, ...],
                               masks: tuple[str, ...],
                               case_names: tuple[str, ...],
                               fixed: str,
                               dataset_name: str,
                               output_folder: str = "deepmedic_workspace/preprocessed/"):
    """
    A function to preprocess ISLES dataset images for deep learning analysis.

    Parameters:
    - flair_images: tuple of strings, paths to FLAIR images
    - dwi_images: tuple of strings, paths to DWI images
    - masks: tuple of strings, paths to masks
    - case_names: tuple of strings, names of the cases
    - fixed: str, either 'flair' or 'dwi'
    - dataset_name: str, name of the dataset
    - output_folder: str, path to the output folder (default is "deepmedic_workspace/preprocessed/")

    Returns:
    - None
    """
    assert fixed in ["flair", "dwi"], "fixed must be either 'flair' or 'dwi'"
    df_cases = pd.DataFrame()
    df_components = pd.DataFrame()

    N = len(flair_images)
    for i, (flair_image, dwi_image, fixed_mask, name) in enumerate(zip(flair_images, dwi_images, masks, case_names)):
        # print progress
        print(f"{dataset_name}: Processing {name} ({i+1}/{N})...")

        # load images
        flair = nib.load(flair_image)
        dwi = nib.load(dwi_image)
        mask = nib.load(fixed_mask)
        orig_mask = nib.load(fixed_mask)

        mask = nib.Nifti1Image(mask.get_fdata().round().astype(np.int8), affine=mask.affine)

        # compute components
        components = cc3d.connected_components(mask.get_fdata(), connectivity=26)
        for component in np.unique(components)[1:]:
            if fixed == "flair":
                volume = voxel_count_to_volume_ml(np.count_nonzero(components == component), flair.header.get_zooms())
            elif fixed == "dwi":
                volume = voxel_count_to_volume_ml(np.count_nonzero(components == component), dwi.header.get_zooms())
            stats = {
                "name": name,
                "volume_ml": volume
            }
            df_components = pd.concat([df_components, pd.DataFrame([stats])])

        # compute volumes
        if fixed == "flair":
            volume = voxel_count_to_volume_ml(np.count_nonzero(mask.get_fdata()), flair.header.get_zooms())
        elif fixed == "dwi":
            volume = voxel_count_to_volume_ml(np.count_nonzero(mask.get_fdata()), dwi.header.get_zooms())

        # save metadata to dict
        stats = {
            "name": name,
            "shape_flair": flair.shape,
            "shape_dwi": dwi.shape,
            "voxel_dim_flair": flair.header.get_zooms(),
            "voxel_dim_dwi": dwi.header.get_zooms(),
            "lesion_volume_ml": volume
        }

        # define ROI mask
        if fixed == "flair":
            ROImask = nib.Nifti1Image((flair.get_fdata() > 0).astype(np.int8), affine=flair.affine)
        elif fixed == "dwi":
            ROImask = nib.Nifti1Image((dwi.get_fdata() > 0).astype(np.int8), affine=dwi.affine)

        stats["bet_mask_volume_ml"] = voxel_count_to_volume_ml(np.count_nonzero(ROImask.get_fdata()), ROImask.header.get_zooms())

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
        
        ROImask = nibabel.processing.resample_from_to(ROImask, flair, order=0)
        mask = nibabel.processing.resample_from_to(mask, flair, order=0)

        # normalize scans within ROI
        n_flair = flair.get_fdata()[ROImask.get_fdata() == 1]
        n_flair = (flair.get_fdata() - n_flair.mean()) / n_flair.std()
        n_dwi = dwi.get_fdata()[ROImask.get_fdata() == 1]
        n_dwi = (dwi.get_fdata() - n_dwi.mean()) / n_dwi.std()
        flair = nib.Nifti1Image(n_flair, affine=flair.affine)
        dwi = nib.Nifti1Image(n_dwi, affine=dwi.affine)

        # check dimensions, mean and std
        assert np.allclose(flair.get_fdata()[ROImask.get_fdata() == 1].mean(), 0) and np.allclose(flair.get_fdata()[ROImask.get_fdata() == 1].std(), 1), \
            f"Mean and std within ROI of flair image are not 0 and 1: {flair.get_fdata()[ROImask.get_fdata() == 1].mean()}, {flair.get_fdata()[ROImask.get_fdata() == 1].std()}"
        assert np.allclose(dwi.get_fdata()[ROImask.get_fdata() == 1].mean(), 0) and np.allclose(dwi.get_fdata()[ROImask.get_fdata() == 1].std(), 1), \
            f"Mean and std within ROI of dwi image are not 0 and 1: {dwi.get_fdata()[ROImask.get_fdata() == 1].mean()}, {dwi.get_fdata()[ROImask.get_fdata() == 1].std()}"
        assert flair.shape == dwi.shape == mask.shape, f"Shapes are not the same: {flair.shape}, {dwi.shape}, {mask.shape}"
        assert np.allclose(flair.affine, dwi.affine), f"FLAIR and DWI have different affines:\n{flair.affine}\n{dwi.affine}"
        assert np.allclose(flair.affine, mask.affine), f"FLAIR and mask have different affines:\n{flair.affine}\n{mask.affine}"
        if not name in ["sub-strokecase0150", "sub-strokecase0151", "sub-strokecase0170"]:
            # skip non-stroke cases
            assert (mask.get_fdata()==1).any(), "Mask is empty"

        # check error after preprocessing
        mask_interpol = nibabel.processing.resample_from_to(mask, orig_mask, order=0)
        stats["dice_after_preprocessing"] = dice_coefficient(orig_mask, mask_interpol)
        df_cases = pd.concat([df_cases, pd.DataFrame([stats])])

        # prepare folder
        if not os.path.exists(f"{output_folder}/{dataset_name}/{name}"):
            os.makedirs(f"{output_folder}/{dataset_name}/{name}")

        # save images
        nib.save(mask, f"{output_folder}/{dataset_name}/{name}/label.nii.gz")
        nib.save(flair, f"{output_folder}/{dataset_name}/{name}/flair.nii.gz")
        nib.save(dwi, f"{output_folder}/{dataset_name}/{name}/dwi.nii.gz")
        nib.save(ROImask, f"{output_folder}/{dataset_name}/{name}/ROImask.nii.gz")

    # save dataframes
    with pd.ExcelWriter(f"{output_folder}/{dataset_name}_metadata.ods") as writer:
        df_cases.to_excel(writer, sheet_name="cases", index=False)
        df_components.to_excel(writer, sheet_name="components", index=False)

if __name__ == "__main__":
    # load datasets
    motol = dataset_loaders.Motol()
    isles2015 = dataset_loaders.ISLES2015()
    isles2022 = dataset_loaders.ISLES2022()

    # run preprocessing for each dataset in parallel
    p1 = multiprocessing.Process(target=deepmedic_preprocess_Motol, 
                                 args=(motol.flairs, motol.dwis, motol.masks, motol.names, motol.BETmasks, "Motol"))

    p2 = multiprocessing.Process(target=deepmedic_preprocess_ISLES,
                                 args=(isles2015.flairs, isles2015.dwis, isles2015.masks, isles2015.names,"flair", "ISLES2015"))

    p3 = multiprocessing.Process(target=deepmedic_preprocess_ISLES,
                                 args=(isles2022.flairs, isles2022.dwis, isles2022.masks, isles2022.names, "dwi", "ISLES2022"))
    
    p1.start()
    p2.start()
    p3.start()

    p1.join()
    p2.join()
    p3.join()