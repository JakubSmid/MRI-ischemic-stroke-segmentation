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

def nnunet_preprocess(flair_images: tuple[str, ...],
                      dwi_images: tuple[str, ...],
                      masks: tuple[str, ...],
                      fixed: str,
                      case_names: tuple[str, ...],
                      BETmasks: tuple[str]|None = None,
                      output_folder: str = "nnUNet_raw/"):
    """
    Preprocesses the input MRI images and masks, computes statistics, performs brain extraction, reshapes images, normalizes scans, calculates dice coefficient, and saves the preprocessed images and statistics to the output folder.

    Args:
    - flair_images (tuple[str, ...]): Paths to the FLAIR images.
    - dwi_images (tuple[str, ...]): Paths to the DWI images.
    - masks (tuple[str, ...]): Paths to the masks.
    - fixed (str): Indicates whether to use 'flair' or 'dwi' as the fixed image.
    - case_names (tuple[str, ...]): Names of the cases.
    - BETmasks (tuple[str]|None, optional): Paths to the brain extraction masks. If BET masks are provided, images have to be co-registered! Defaults to None.
    - output_folder (str, optional): Output folder path. Defaults to "preprocessed/".

    Returns:
    - None
    """
    assert fixed in ["flair", "dwi"], "fixed must be 'flair' or 'dwi'"

    N = len(flair_images)
    for i, (flair_image, dwi_image, fixed_mask, name) in enumerate(zip(flair_images, dwi_images, masks, case_names)):
        # print progress
        name = name.replace("/", "_")
        print(f"Processing {name} ({i+1}/{N})...")

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

        # brain extraction: use HD-BET output or filter values greater than 0
        # If BET masks are provided, images have to be co-registered!
        if BETmasks is not None:
            ROImask = nib.load(BETmasks[i])
            # DeepMedic bug workaround: recasting ROI from uint8 to int8
            ROImask = nib.Nifti1Image(ROImask.get_fdata().astype(np.int8), affine=ROImask.affine)
            flair = nib.Nifti1Image(flair.get_fdata() * ROImask.get_fdata(), affine=flair.affine)
            dwi = nib.Nifti1Image(dwi.get_fdata() * ROImask.get_fdata(), affine=dwi.affine)
            mask = nib.Nifti1Image((mask.get_fdata() * ROImask.get_fdata()).astype(np.int8), affine=mask.affine)
            if ".nrrd" in fixed_mask:
                mask_flair = nib.Nifti1Image((mask_flair.get_fdata() * ROImask.get_fdata()).astype(np.int8), affine=mask_flair.affine)
                mask_dwi = nib.Nifti1Image((mask_dwi.get_fdata() * ROImask.get_fdata()).astype(np.int8), affine=mask_dwi.affine)
        
        # reshape masks
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

        # check dimensions
        assert flair.shape == dwi.shape == mask.shape, f"Shapes are not the same: {flair.shape}, {dwi.shape}, {mask.shape}"

        # prepare folder
        if not os.path.exists(f"{output_folder}/Dataset001_Strokes/imagesTr"):
            os.makedirs(f"{output_folder}/Dataset001_Strokes/imagesTr")
        if not os.path.exists(f"{output_folder}/Dataset001_Strokes/labelsTr"):
            os.makedirs(f"{output_folder}/Dataset001_Strokes/labelsTr")
        if ".nrrd" in fixed_mask:
            if not os.path.exists(f"{output_folder}/Dataset002_FlairMasks/labelsTr"):
                os.makedirs(f"{output_folder}/Dataset002_FlairMasks/labelsTr")
            if not os.path.exists(f"{output_folder}/Dataset003_DwiMasks/labelsTr"):
                os.makedirs(f"{output_folder}/Dataset003_DwiMasks/labelsTr")

        # save images
        # modalities: imagesTr/{CASE_IDENTIFIER}_{XXXX}.{FILE_ENDING}
        # masks: labelsTr/{CASE_IDENTIFER}.{FILE_ENDING}
        nib.save(mask, f"{output_folder}/Dataset001_Strokes/labelsTr/{name}.nii.gz")
        nib.save(flair, f"{output_folder}/Dataset001_Strokes/imagesTr/{name}_0000.nii.gz")
        nib.save(dwi, f"{output_folder}/Dataset001_Strokes/imagesTr/{name}_0001.nii.gz")
        if ".nrrd" in fixed_mask:
            nib.save(mask_flair, f"{output_folder}/Dataset002_FlairMasks/labelsTr/{name}.nii.gz")
            nib.save(mask_dwi, f"{output_folder}/Dataset003_DwiMasks/labelsTr/{name}.nii.gz")

if __name__ == "__main__":
    # load datasets
    motol = dataset_loaders.Motol()
    isles2015 = dataset_loaders.ISLES2015()
    isles2022 = dataset_loaders.ISLES2022()

    # run preprocessing for each dataset in parallel
    motol_p = multiprocessing.Process(target=nnunet_preprocess, 
                                      args=(motol.flairs, motol.dwis, motol.masks, "flair", motol.names, motol.BETmasks))

    isles15_p = multiprocessing.Process(target=nnunet_preprocess,
                                        args=(isles2015.flairs, isles2015.dwis, isles2015.masks, "flair", isles2015.names))

    isles22_p = multiprocessing.Process(target=nnunet_preprocess,
                                        args=(isles2022.flairs, isles2022.dwis, isles2022.masks, "dwi", isles2022.names))
    
    motol_p.start()
    isles15_p.start()
    isles22_p.start()

    motol_p.join()
    isles15_p.join()
    isles22_p.join()