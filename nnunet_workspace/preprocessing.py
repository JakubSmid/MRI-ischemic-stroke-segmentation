import nibabel as nib
import nibabel.processing
import ants
import os
import numpy as np
import multiprocessing

import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from datasets.utils import *
import datasets.dataset_loaders as dataset_loaders

def nnunet_preprocess_Motol(flair_images: tuple[str, ...],
                            dwi_images: tuple[str, ...],
                            masks: tuple[str, ...],
                            case_names: tuple[str, ...],
                            BETmasks: tuple[str, ...],
                            dataset_name: str = "Motol",
                            output_folder: str = "nnunet_workspace/nnUNet_raw/"):
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
    N = len(flair_images)
    for i, (flair_image, dwi_image, fixed_mask, BET_mask, name) in enumerate(zip(flair_images, dwi_images, masks, BETmasks, case_names)):
        # print progress
        print(f"{dataset_name}: Processing {name} ({i+1}/{N})...")

        # load images and masks
        flair = nib.load(flair_image)
        dwi = nib.load(dwi_image)
        mask_flair, mask_dwi = nrrd_to_nifti(fixed_mask)
        mask = np.logical_or(mask_flair.get_fdata(), mask_dwi.get_fdata())
        mask = nib.Nifti1Image(mask.astype(np.int8), affine=mask_flair.affine)

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
        # assume BET masks are co-registered to FLAIR
        ROImask = nib.load(BET_mask)

        # reshape masks
        ROImask = nibabel.processing.resample_from_to(ROImask, flair, order=0)
        mask = nibabel.processing.resample_from_to(mask, flair, order=0)
        
        # apply BET mask
        flair = nib.Nifti1Image(flair.get_fdata() * ROImask.get_fdata(), affine=flair.affine)
        dwi = nib.Nifti1Image(dwi.get_fdata() * ROImask.get_fdata(), affine=dwi.affine)
        mask = nib.Nifti1Image((mask.get_fdata() * ROImask.get_fdata()).astype(np.int8), affine=mask.affine)

        # check dimensions
        assert flair.shape == dwi.shape == mask.shape, f"Shapes are not the same: {flair.shape}, {dwi.shape}, {mask.shape}"
        assert np.allclose(flair.affine, dwi.affine), f"FLAIR and DWI have different affines:\n{flair.affine}\n{dwi.affine}"
        assert np.allclose(flair.affine, mask.affine), f"FLAIR and mask have different affines:\n{flair.affine}\n{mask.affine}"
        assert (mask.get_fdata()==1).any(), "Mask is empty"

        # prepare folders
        if not os.path.exists(f"{output_folder}/Dataset001_Strokes/imagesTr"):
            os.makedirs(f"{output_folder}/Dataset001_Strokes/imagesTr")
        if not os.path.exists(f"{output_folder}/Dataset001_Strokes/labelsTr"):
            os.makedirs(f"{output_folder}/Dataset001_Strokes/labelsTr")

        # save images
        # modalities: imagesTr/{CASE_IDENTIFIER}_{XXXX}.{FILE_ENDING}
        # masks: labelsTr/{CASE_IDENTIFER}.{FILE_ENDING}
        nib.save(mask, f"{output_folder}/Dataset001_Strokes/labelsTr/{name}.nii.gz")
        nib.save(flair, f"{output_folder}/Dataset001_Strokes/imagesTr/{name}_0000.nii.gz")
        nib.save(dwi, f"{output_folder}/Dataset001_Strokes/imagesTr/{name}_0001.nii.gz")

def nnunet_preprocess_ISLES(flair_images: tuple[str, ...],
                            dwi_images: tuple[str, ...],
                            masks: tuple[str, ...],
                            case_names: tuple[str, ...],
                            fixed: str,
                            dataset_name: str,
                            output_folder: str = "nnunet_workspace/nnUNet_raw/"):
    """
    Preprocesses the input MRI images and masks, computes statistics, performs brain extraction, reshapes images, normalizes scans, calculates dice coefficient, and saves the preprocessed images and statistics to the output folder.

    Args:
    - flair_images (tuple[str, ...]): Paths to the FLAIR images.
    - dwi_images (tuple[str, ...]): Paths to the DWI images.
    - masks (tuple[str, ...]): Paths to the masks..
    - case_names (tuple[str, ...]): Names of the cases.
    - fixed (str): Indicates whether to use 'flair' or 'dwi' as the fixed image.
    - dataset_name: str, name of the dataset
    - output_folder (str, optional): Output folder path. Defaults to "preprocessed/".

    Returns:
    - None
    """
    assert fixed in ["flair", "dwi"], "fixed must be 'flair' or 'dwi'"

    N = len(flair_images)
    for i, (flair_image, dwi_image, fixed_mask, name) in enumerate(zip(flair_images, dwi_images, masks, case_names)):
        # print progress
        print(f"{dataset_name}: Processing {name} ({i+1}/{N})...")

        # load images
        flair = nib.load(flair_image)
        dwi = nib.load(dwi_image)
        mask = nib.load(fixed_mask)

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

        mask = nibabel.processing.resample_from_to(mask, flair, order=0)
        mask = nib.Nifti1Image(mask.get_fdata().round().astype(np.int8), affine=mask.affine)

        # check dimensions
        assert flair.shape == dwi.shape == mask.shape, f"Shapes are not the same: {flair.shape}, {dwi.shape}, {mask.shape}"
        assert np.allclose(flair.affine, dwi.affine), f"FLAIR and DWI have different affines:\n{flair.affine}\n{dwi.affine}"
        assert np.allclose(flair.affine, mask.affine), f"FLAIR and mask have different affines:\n{flair.affine}\n{mask.affine}"
        assert (mask.get_fdata()==1).any(), "Mask is empty"

        # prepare folder
        if not os.path.exists(f"{output_folder}/Dataset001_Strokes/imagesTr"):
            os.makedirs(f"{output_folder}/Dataset001_Strokes/imagesTr")
        if not os.path.exists(f"{output_folder}/Dataset001_Strokes/labelsTr"):
            os.makedirs(f"{output_folder}/Dataset001_Strokes/labelsTr")

        # save images
        # modalities: imagesTr/{CASE_IDENTIFIER}_{XXXX}.{FILE_ENDING}
        # masks: labelsTr/{CASE_IDENTIFER}.{FILE_ENDING}
        nib.save(mask, f"{output_folder}/Dataset001_Strokes/labelsTr/{name}.nii.gz")
        nib.save(flair, f"{output_folder}/Dataset001_Strokes/imagesTr/{name}_0000.nii.gz")
        nib.save(dwi, f"{output_folder}/Dataset001_Strokes/imagesTr/{name}_0001.nii.gz")

if __name__ == "__main__":
    # load datasets
    motol = dataset_loaders.Motol()
    isles2015 = dataset_loaders.ISLES2015()
    isles2022 = dataset_loaders.ISLES2022()

    # run preprocessing for each dataset in parallel
    motol_p = multiprocessing.Process(target=nnunet_preprocess_Motol,
                                      args=(motol.flairs, motol.dwis, motol.masks, motol.names, motol.BETmasks))

    isles15_p = multiprocessing.Process(target=nnunet_preprocess_ISLES,
                                        args=(isles2015.flairs, isles2015.dwis, isles2015.masks, isles2015.names, "flair", "ISLES2015"))

    isles22_p = multiprocessing.Process(target=nnunet_preprocess_ISLES,
                                        args=(isles2022.flairs, isles2022.dwis, isles2022.masks, isles2022.names, "dwi", "ISLES2022"))
    
    motol_p.start()
    isles15_p.start()
    isles22_p.start()

    motol_p.join()
    isles15_p.join()
    isles22_p.join()