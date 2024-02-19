import os
import nibabel as nib
import nibabel.processing
import numpy as np
import pandas as pd
import cc3d

def voxel_count_to_volume(voxel_count: int, voxel_size: tuple[float, float, float]) -> float:
    return voxel_count * np.prod(voxel_size) / 1000

statistics = {
    "id": [],
    "shape": [],
    "voxel_dim_flair": [],
    "voxel_dim_dwi": [],
    "flair_volume_ml": [],
    "number_of_flair_components": [],
}

for patient_folder in range(1,29):
    print(f"Processing patient {patient_folder}/28")

    # load folders with scans
    scan_folders = os.listdir(f"datasets/SISS2015_Training/{patient_folder}")
    flair = next(filter(lambda x: "Flair" in x, scan_folders))
    dwi = next(filter(lambda x: "DWI" in x, scan_folders))
    flair_mask = next(filter(lambda x: "OT" in x, scan_folders))

    # load scans
    flair = nib.load(f"datasets/SISS2015_Training/{patient_folder}/{flair}/{flair}.nii")
    dwi = nib.load(f"datasets/SISS2015_Training/{patient_folder}/{dwi}/{dwi}.nii")
    flair_mask = nib.load(f"datasets/SISS2015_Training/{patient_folder}/{flair_mask}/{flair_mask}.nii")

    # compute statistics
    flair_components = cc3d.connected_components(flair_mask.get_fdata(), connectivity=26)
    flair_volume = voxel_count_to_volume(np.count_nonzero(flair_mask.get_fdata()), flair.header.get_zooms())

    # add statistics
    statistics["id"].append(f"{patient_folder}")
    statistics["shape"].append(flair.shape)
    statistics["voxel_dim_flair"].append(flair.header.get_zooms())
    statistics["voxel_dim_dwi"].append(dwi.header.get_zooms())
    statistics["flair_volume_ml"].append(flair_volume)
    statistics["number_of_flair_components"].append(flair_components.max())

    # resample to 1x1x1 mm voxel size
    flair = nibabel.processing.conform(flair, voxel_size=(1,1,1), out_shape=(200,200,200))
    dwi = nibabel.processing.conform(dwi, voxel_size=(1,1,1), out_shape=(200,200,200))
    flair_mask = nibabel.processing.conform(flair_mask, voxel_size=(1,1,1), out_shape=(200,200,200))

    # normalize
    normalized_flair = (flair.get_fdata() - flair.get_fdata().mean()) / flair.get_fdata().std()
    normalized_dwi = (dwi.get_fdata() - dwi.get_fdata().mean()) / dwi.get_fdata().std()
    flair = nib.Nifti1Image(normalized_flair, flair.affine, dtype=flair.get_data_dtype())
    dwi = nib.Nifti1Image(normalized_dwi, dwi.affine, dtype=dwi.get_data_dtype())

    # check dimensions, mean and std
    assert flair.shape == dwi.shape == flair_mask.shape, f"Shapes are not the same: {flair.shape}, {dwi.shape}, {flair_mask.shape}"
    assert np.allclose(flair.get_fdata().mean(), 0) and np.allclose(flair.get_fdata().std(), 1), f"Flair is not zero mean and unit variance: {flair.get_fdata().mean()}, {flair.get_fdata().std()}"
    assert np.allclose(dwi.get_fdata().mean(), 0) and np.allclose(dwi.get_fdata().std(), 1), f"Dwi is not zero mean and unit variance: {dwi.get_fdata().mean()}, {dwi.get_fdata().std()}"

    # prepare folder
    if not os.path.exists(f"preprocessed/DeepMedic/SISS2015/{patient_folder}/"):
        os.makedirs(f"preprocessed/DeepMedic/SISS2015/{patient_folder}/")

    # save images
    nib.save(flair_mask, f"preprocessed/DeepMedic/SISS2015/{patient_folder}/flair_mask.nii")
    nib.save(flair, f"preprocessed/DeepMedic/SISS2015/{patient_folder}/flair.nii")
    nib.save(dwi, f"preprocessed/DeepMedic/SISS2015/{patient_folder}/dwi.nii")

# save statistics
pd.DataFrame(statistics).to_excel("preprocessed/DeepMedic/SISS2015/statistics.ods", index=False)