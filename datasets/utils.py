import numpy as np
import nibabel as nib
import nrrd

def dice_coefficient(y_true: nib.Nifti1Image, y_pred: nib.Nifti1Image) -> float:
    """
    Calculate the Dice coefficient between two Nifti1Image masks.
    
    Args:
    - y_true: a Nifti1Image object representing the ground truth segmentation
    - y_pred: a Nifti1Image object representing the predicted segmentation
    
    Returns:
    - float: the Dice coefficient
    """
    y_true = y_true.get_fdata()
    y_pred = y_pred.get_fdata()
    intersection = np.count_nonzero(y_true * y_pred)
    if y_pred.sum() == 0 and y_true.sum() == 0:
        return 1
    elif intersection == 0:
        return 0
    return 2 * intersection / (np.sum(y_pred) + np.sum(y_true))

def voxel_count_to_volume_ml(voxel_count: int, voxel_zooms: tuple[float, float, float]) -> float:
    """
    Calculate the volume in milliliters based on the voxel count and voxel zooms.

    Args:
    - voxel_count (int): The number of voxels.
    - voxel_zooms (tuple[float, float, float]): The size of each voxel in millimeters in x, y, and z dimensions.

    Returns:
    - float: The volume in milliliters.
    """
    return voxel_count * np.prod(voxel_zooms) / 1000

def trim_zero_padding(MRI_image: nib.Nifti1Image) -> nib.Nifti1Image:
    """
    Trim zero padding from the input Nifti1Image and return the trimmed image.
    
    Args:
    - MRI_image: a Nifti1Image object representing the MRI image
    
    Returns:
    - nib.Nifti1Image: the trimmed Nifti1Image object
    """
    xs, ys, zs = np.where(MRI_image.get_fdata() != 0)
    # if no nonzero voxels found return original image
    if len(xs) != 0:
        MRI_image = MRI_image.slicer[min(xs):max(xs)+1,min(ys):max(ys)+1,min(zs):max(zs)+1]
    return MRI_image

def nrrd_to_nifti(nrrd_path: str, affine: np.ndarray) -> tuple[nib.Nifti1Image, nib.Nifti1Image]:
    """
    Convert the data from an nrrd file to two Nifti1Image objects representing the FLAIR and DWI segments.
    
    Args:
    - nrrd_path (str): The path to the nrrd file.
    - affine (np.ndarray): The affine transformation matrix for masks.
        
    Returns:
    - tuple[nib.Nifti1Image, nib.Nifti1Image]: A tuple containing the Nifti1Image objects for the FLAIR and DWI segments.
    """

    # load nrrd
    data, header = nrrd.read(nrrd_path)

    # find the FLAIR and DWI segments
    for key, value in header.items():
        if "FLAIR" in str(value).upper():
            flair_segment = key.split("_")[0]
        if "DWI" in str(value).upper():
            dwi_segment = key.split("_")[0]
    
    flair_layer = header[flair_segment + "_Layer"]
    flair_value = header[flair_segment + "_LabelValue"]
    
    dwi_layer = header[dwi_segment + "_Layer"]
    dwi_value = header[dwi_segment + "_LabelValue"]

    # extract masks from nrrd
    flair_mask = np.where(data == int(flair_value), 1, 0)[int(flair_layer), :, :, :].astype(np.int8)
    dwi_mask = np.where(data == int(dwi_value), 1, 0)[int(dwi_layer), :, :, :].astype(np.int8)

    # return nifti images
    nifti_flair = nib.Nifti1Image(flair_mask, affine=affine)
    nifti_dwi = nib.Nifti1Image(dwi_mask, affine=affine)

    return nifti_flair, nifti_dwi