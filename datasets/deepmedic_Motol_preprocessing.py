import os
import nibabel as nib
import nibabel.processing
import numpy as np
import nrrd
import pandas as pd
import cc3d

def voxel_count_to_volume(voxel_count: int, voxel_size: tuple[float, float, float]) -> float:
    return voxel_count * np.prod(voxel_size) / 1000

def nrrd_to_nifti(nrrd_path: str, affine: np.ndarray) -> tuple[nib.Nifti1Image, nib.Nifti1Image]:
    """
    nrrd_path: path to nrrd file
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

    nifti_flair.set_data_dtype("mask")
    nifti_dwi.set_data_dtype("mask")

    return nifti_flair, nifti_dwi

statistics = {
    "id": [],
    "shape": [],
    "voxel_dim_flair": [],
    "voxel_dim_dwi": [],
    "flair_volume_ml": [],
    "dwi_volume_ml": [],
    "number_of_flair_components": [],
    "number_of_dwi_components": []
}

for patient_folder in os.listdir('datasets/Motol'):
    print(f"Processing {patient_folder}...")
    for anat in os.listdir(f'datasets/Motol/{patient_folder}'):
        # skip corrupted data
        corrupted = ["Anat_20211008","Anat_20230109"]
        if anat in corrupted:
            continue

        # load scans
        flair = nib.load(f"datasets/Motol/{patient_folder}/{anat}/rFlair.nii")
        dwi = nib.load(f"datasets/Motol/{patient_folder}/{anat}/rDWI2.nii")
        flair_mask, dwi_mask = nrrd_to_nifti(f"datasets/Motol/{patient_folder}/{anat}/Leze_FLAIR_DWI2.nrrd", flair.affine)

        # compute statistics
        flair_components = cc3d.connected_components(flair_mask.get_fdata(), connectivity=26)
        dwi_components = cc3d.connected_components(dwi_mask.get_fdata(), connectivity=26)
        flair_volume = voxel_count_to_volume(np.count_nonzero(flair_mask.get_fdata()), flair.header.get_zooms())
        dwi_volume = voxel_count_to_volume(np.count_nonzero(dwi_mask.get_fdata()), dwi.header.get_zooms())

        # save statistics
        statistics["id"].append(f"{patient_folder}/{anat}")
        statistics["shape"].append(flair.shape)
        statistics["voxel_dim_flair"].append(flair.header.get_zooms())
        statistics["voxel_dim_dwi"].append(dwi.header.get_zooms())
        statistics["flair_volume_ml"].append(flair_volume)
        statistics["dwi_volume_ml"].append(dwi_volume)
        statistics["number_of_flair_components"].append(flair_components.max())
        statistics["number_of_dwi_components"].append(dwi_components.max())

        # resample to 1x1x1 mm voxel size
        flair = nibabel.processing.conform(flair, voxel_size=(1,1,1), out_shape=(200,200,200))
        dwi = nibabel.processing.conform(dwi, voxel_size=(1,1,1), out_shape=(200,200,200))
        flair_mask = nibabel.processing.conform(flair_mask, voxel_size=(1,1,1), out_shape=(200,200,200))
        dwi_mask = nibabel.processing.conform(dwi_mask, voxel_size=(1,1,1), out_shape=(200,200,200))

        # normalize
        normalized_flair = (flair.get_fdata() - flair.get_fdata().mean()) / flair.get_fdata().std()
        normalized_dwi = (dwi.get_fdata() - dwi.get_fdata().mean()) / dwi.get_fdata().std()
        flair = nib.Nifti1Image(normalized_flair, flair.affine, dtype=flair.get_data_dtype())
        dwi = nib.Nifti1Image(normalized_dwi, dwi.affine, dtype=dwi.get_data_dtype())

        # check dimensions, mean and std
        assert flair.shape == dwi.shape == flair_mask.shape == dwi_mask.shape, f"Shapes are not the same: {flair.shape}, {dwi.shape}, {flair_mask.shape}, {dwi_mask.shape}"
        assert np.allclose(flair.get_fdata().mean(), 0) and np.allclose(flair.get_fdata().std(), 1), f"Flair is not zero mean and unit variance: {flair.get_fdata().mean()}, {flair.get_fdata().std()}"
        assert np.allclose(dwi.get_fdata().mean(), 0) and np.allclose(dwi.get_fdata().std(), 1), f"Dwi is not zero mean and unit variance: {dwi.get_fdata().mean()}, {dwi.get_fdata().std()}"

        # prepare folder
        if not os.path.exists(f"preprocessed/DeepMedic/Motol/{patient_folder}/{anat}/"):
            os.makedirs(f"preprocessed/DeepMedic/Motol/{patient_folder}/{anat}/")

        # save images
        nib.save(flair_mask, f"preprocessed/DeepMedic/Motol/{patient_folder}/{anat}/flair_mask.nii")
        nib.save(dwi_mask, f"preprocessed/DeepMedic/Motol/{patient_folder}/{anat}/dwi_mask.nii")
        nib.save(flair, f"preprocessed/DeepMedic/Motol/{patient_folder}/{anat}/flair.nii")
        nib.save(dwi, f"preprocessed/DeepMedic/Motol/{patient_folder}/{anat}/dwi.nii")

# save statistics
pd.DataFrame(statistics).to_excel("preprocessed/DeepMedic/Motol/statistics.ods", index=False)
