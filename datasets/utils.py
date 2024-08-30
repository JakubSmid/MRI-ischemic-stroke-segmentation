import numpy as np
import ants
import nrrd

def subtract_masks(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Generate a new array by performing a logical AND operation between `x` and the negation of `y`.
    
    Parameters:
        x (np.ndarray): The first input array.
        y (np.ndarray): The second input array.
        
    Returns:
        np.ndarray: The resulting array.
    """
    return np.logical_and(x, np.logical_not(y)) * 1

def dice_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the Dice coefficient between two Nifti1Image or numpy masks.
    
    Parameters:
        y_true (np.ndarray): a numpy array representing the ground truth segmentation
        y_pred (np.ndarray): a numpy array representing the predicted segmentation
    
    Returns:
        float: the Dice coefficient
    """
    intersection = np.count_nonzero(y_true * y_pred)
    if y_pred.sum() == 0 and y_true.sum() == 0:
        return 1
    elif intersection == 0:
        return 0
    return 2 * intersection / (np.sum(y_pred) + np.sum(y_true))

def voxel_count_to_volume_ml(voxel_count: int, voxel_zooms: tuple[float, float, float]) -> float:
    """
    Calculate the volume in milliliters based on the voxel count and voxel zooms.

    Parameters:
        voxel_count (int): The number of voxels.
        voxel_zooms (tuple[float, float, float]): The size of each voxel in millimeters in x, y, and z dimensions.

    Returns:
        float: The volume in milliliters.
    """
    return voxel_count * np.prod(voxel_zooms) / 1000

def load_nrrd(nrrd_path: str) -> list[ants.ants_image.ANTsImage]:
    """
    Load an nrrd file from Motol dataset and extract FLAIR and DWI segmentations. 
    Check that there is exactly one FLAIR and one DWI segmentation in the header.
    Create ANTs images from the extracted masks. 

    Parameters:
        nrrd_path (str): The file path to the nrrd file.

    Returns:
        list (ants.ants_image.ANTsImage): Two ANTs images representing the FLAIR and DWI segmentations.
    """
    # load nrrd
    data, header = nrrd.read(nrrd_path)
    assert header["space"] == "left-posterior-superior", f"Space should be 'left-posterior-superior', but it is {header['space']}"
    
    # find the FLAIR and DWI segmentations
    dwi = 0
    flair = 0
    for key, value in header.items():
        if "FLAIR" in str(value).upper():
            flair_segment = key.split("_")[0]
            flair += 1
        if "DWI" in str(value).upper():
            dwi_segment = key.split("_")[0]
            dwi += 1
    assert flair == 1 and dwi == 1, f"{nrrd_path}: There should be exactly one FLAIR and one DWI segmentation, but there are {flair} FLAIR segmentations and {dwi} DWI segmentations"
    
    flair_layer = header[flair_segment + "_Layer"]
    flair_value = header[flair_segment + "_LabelValue"]
    
    dwi_layer = header[dwi_segment + "_Layer"]
    dwi_value = header[dwi_segment + "_LabelValue"]

    # extract masks from nrrd
    flair_mask = np.where(data == int(flair_value), 1, 0)[int(flair_layer), :, :, :].astype(np.uint32)
    dwi_mask = np.where(data == int(dwi_value), 1, 0)[int(dwi_layer), :, :, :].astype(np.uint32)

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

def invert_SyN_registration(image: ants.ants_image.ANTsImage, warp_file: str, affine_file: str) -> ants.ants_image.ANTsImage:
    """
    Inverts the SyN registration for the given image using the provided warp and affine files.

    Parameters:
        image (ants.ants_image.ANTsImage): Image to invert the registration for.
        warp_file (str): The transformation warp file.
        affine_file (str): The affine transformation file.

    Returns:
        ants.ants_image.ANTsImage: Applied inversion transformation of SyN to the image.
    """
    warp = ants.image_read(warp_file).apply(lambda x: -x)
    warptx = ants.transform_from_displacement_field(warp)
    affinetx = ants.read_transform(affine_file).invert()
    
    inverted = warptx.apply_to_image(image)
    inverted = affinetx.apply_to_image(inverted)
    return inverted

def apply_transform_to_label(label: ants.ants_image.ANTsImage, transform: ants.ANTsTransform, reference: ants.ants_image.ANTsImage = None) -> ants.ants_image.ANTsImage:
    """
    Apply a transformation to the input label image.

    Parameters:
        label (ants.ants_image.ANTsImage): The input label image.
        transform (ants.ANTsTransform): The transformation to apply.
        reference (ants.ants_image.ANTsImage, optional): The reference space for transformation. Defaults to None.

    Returns:
        ants.ants_image.ANTsImage: The transformed label image as uint32 in reference space.
    """
    # interpolation="genericLabel" or "genericlabel" throws: ITK ERROR: ResampleImageFilter(0x4572160): Interpolator not set
    transformed = transform.apply_to_image(label.astype("float32"), reference, interpolation="linear")
    return transformed.new_image_like(transformed.numpy().round().astype(np.uint32))

def resample_label_to_target(label: ants.ants_image.ANTsImage, target_image: ants.ants_image.ANTsImage) -> ants.ants_image.ANTsImage:
    """
    Resamples the input label image to the target image.

    Parameters:
        label (ants.ants_image.ANTsImage): The input label image.
        target_image (ants.ants_image.ANTsImage): The target image.

    Returns:
        ants.ants_image.ANTsImage: The resampled label image.
    """
    # interpolation="genericlabel" gives output with floating point which results in smaller label
    resampled = ants.resample_image_to_target(label.astype("float32"), target_image, interpolation="linear")
    return resampled.new_image_like(resampled.numpy().round().astype(np.uint32))