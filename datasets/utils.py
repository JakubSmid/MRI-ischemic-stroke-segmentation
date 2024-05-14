import numpy as np
import nibabel as nib
import ants
import nrrd

def subtract_masks(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Generate a new array by performing a logical AND operation between `x` and the negation of `y`.
    
    Args:
        x (np.ndarray): The first input array.
        y (np.ndarray): The second input array.
        
    Returns:
        np.ndarray: The resulting array.
    """
    return np.logical_and(x, np.logical_not(y)) * 1

def dice_coefficient(y_true: nib.Nifti1Image|np.ndarray, y_pred: nib.Nifti1Image|np.ndarray) -> float:
    """
    Calculate the Dice coefficient between two Nifti1Image or numpy masks.
    
    Args:
    - y_true: a numpy array or a Nifti1Image object representing the ground truth segmentation
    - y_pred: a numpy array or a Nifti1Image object representing the predicted segmentation
    
    Returns:
    - float: the Dice coefficient
    """
    if type(y_true) == nib.Nifti1Image and type(y_pred) == nib.Nifti1Image:
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

def load_nrrd(nrrd_path: str) -> list[ants.ANTsImage]:
    """
    Load an nrrd file and extract FLAIR and DWI segmentations. 
    Check that there is exactly one FLAIR and one DWI segmentation in the header.
    Create ANTs images from the extracted masks. 

    Parameters:
    nrrd_path (str): The file path to the nrrd file.

    Returns:
    list[ants.ANTsImage]: Two ANTs images representing the FLAIR and DWI segmentations.
    """
    # load nrrd
    data, header = nrrd.read(nrrd_path)
    assert header["space"] == "left-posterior-superior", f"Space should be 'left-posterior-superior', but it is {header['space']}"
    
    # check multiple FLAIR and DWI segmentations
    dwi = 0
    flair = 0
    for v in header.values():
        if "DWI" in str(v).upper():
            dwi += 1
        if "FLAIR" in str(v).upper():
            flair += 1
    assert dwi == 1 and flair == 1, f"{nrrd_path}: There should be exactly one FLAIR and one DWI segmentation, but there are {flair} FLAIR segmentations and {dwi} DWI segmentations"

    # find the FLAIR and DWI segmentations
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
    flair_mask = np.where(data == int(flair_value), 1, 0)[int(flair_layer), :, :, :].astype(np.uint8)
    dwi_mask = np.where(data == int(dwi_value), 1, 0)[int(dwi_layer), :, :, :].astype(np.uint8)

    affine = np.zeros((4,4))
    affine[:3, :3] = header["space directions"][1:].T
    affine[3, 3] = 1
    affine[:3, 3] = header["space origin"]

    # decompose affine
    origin = affine[:3, 3]
    spacing = np.linalg.norm(affine[:3,:3], axis=0)
    direction = affine[:3, :3] / spacing

    # create ANTs images
    ants_flair = ants.from_numpy(flair_mask, origin=origin.tolist(), direction=direction.tolist(), spacing=spacing.tolist())
    ants_dwi = ants.from_numpy(dwi_mask, origin=origin.tolist(), direction=direction.tolist(), spacing=spacing.tolist())

    return ants_flair, ants_dwi

def invert_SyN_registration(image: ants.ANTsImage, warp_file: str, affine_file: str) -> ants.ANTsImage:
    """
    Inverts the SyN registration for the given image using the provided warp and affine files.

    Parameters:
        image (ants.ANTsImage): The image to be registered.
        warp_file (str): The warp file for the registration.
        affine_file (str): The affine file for the registration.

    Returns:
        ants.ANTsImage: The inverted image after applying the SyN registration.
    """
    warp = ants.image_read(warp_file).apply(lambda x: -x)
    warptx = ants.transform_from_displacement_field(warp)
    affinetx = ants.read_transform(affine_file).invert()
    
    inverted = warptx.apply_to_image(image)
    inverted = affinetx.apply_to_image(inverted)
    return inverted

def apply_transform_to_label(label: ants.ANTsImage, transform: ants.ANTsTransform, reference: ants.ANTsImage = None) -> ants.ANTsImage:
    """
    Apply a transformation to the input label image.

    Parameters:
        label (ants.ANTsImage): The input label image.
        transform (ants.ANTsTransform): The transformation to apply.
        reference (ants.ANTsImage, optional): The reference space for transformation. Defaults to None.

    Returns:
        ants.ANTsImage: The transformed label image as uint32 in reference space.
    """
    transformed = transform.apply_to_image(label, reference, interpolation="linear")
    return transformed.new_image_like(transformed.numpy().round().astype(np.uint32))