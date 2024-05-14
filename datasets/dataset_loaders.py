import os
import ants
import numpy as np
from dataclasses import dataclass
from .utils import load_nrrd, apply_transform_to_label

@dataclass
class Subject():
    name: str
    flair: str | ants.ANTsImage
    dwi: str | ants.ANTsImage
    label: str | ants.ANTsImage
    labeled_modality: str = "flair"
    BETmask: str | ants.ANTsImage = None
    
    def normalize(self):
        """
        Normalizes the flair and dwi images of the subject by subtracting the mean and dividing by the standard deviation.
        """
        assert self.is_loaded(), f"Subject {self.name} is not loaded"
        data = self.flair.numpy()
        self.flair = self.flair.new_image_like((data-data.mean()) / data.std())

        data = self.dwi.numpy()
        self.dwi = self.dwi.new_image_like((data-data.mean()) / data.std())

    def load_data(self,
                  transform_dwi_to_flair=True,
                  transform_to_mni=False,
                  load_label=True,
                  target_shape=(200, 200, 200),
                  target_spacing=(1, 1, 1)):
        """
        Loads and processes images for the Subject, including reading images, applying transformations and resampling to target shape.
        
        Parameters:
            transform_dwi_to_flair (bool): Whether to transform DWI to FLAIR space.
            transform_to_mni (bool): Whether to transform to MNI space.
            load_label (bool): Whether to load the label.
            target_shape (tuple | None): Target shape of the images. If None, the images are not resampled.
            target_spacing (tuple): Target spacing of the images.
        """
        
        assert not self.is_loaded(), f"Subject {self.name} is already loaded"
        
        # save paths
        self._subj_paths = [self.flair, self.dwi, self.label, self.BETmask]

        # load images
        self.flair = ants.image_read(self.flair)
        self.dwi = ants.image_read(self.dwi)
        
        if load_label:
            if ".nrrd" in self.label:
                self.label = load_nrrd(self.label)
                new_label = np.logical_or(self.label[0].numpy(), self.label[1].numpy()).astype(np.uint8)
                self.label = self.label[0].new_image_like(new_label)
            else:
                self.label = ants.image_read(self.label)

        if self.BETmask:
            self.BETmask = ants.image_read(self.BETmask).astype("float32")
        else:
            self.BETmask = self.flair.new_image_like((self.flair.numpy() != 0).astype("float32"))

        if transform_dwi_to_flair:
            dwi_to_flair = ants.read_transform(self.transform_dwi_to_flair)
            self.dwi = dwi_to_flair.apply_to_image(self.dwi, self.flair)
            if self.labeled_modality == "dwi" and load_label:
                self.label = apply_transform_to_label(self.label, dwi_to_flair, self.flair)

        if transform_to_mni:
            template_mni = ants.image_read("datasets/template_flair_mni.nii.gz")
            warp = ants.transform_from_displacement_field(ants.image_read(self.transform_flair_to_mni[0]))
            affine = ants.read_transform(self.transform_flair_to_mni[1])

            self.flair = affine.apply_to_image(self.flair, template_mni)
            self.flair = warp.apply_to_image(self.flair, template_mni)

            self.dwi = affine.apply_to_image(self.dwi, template_mni)
            self.dwi = warp.apply_to_image(self.dwi, template_mni)

            self.BETmask = apply_transform_to_label(self.BETmask, affine, template_mni)
            self.BETmask = apply_transform_to_label(self.BETmask, warp, template_mni).astype("float32")

            self.label = apply_transform_to_label(self.label, affine, template_mni)
            self.label = apply_transform_to_label(self.label, warp, template_mni)

        # brain extraction
        self.flair = ants.mask_image(self.flair, self.BETmask)
        self.dwi = ants.mask_image(self.dwi, self.BETmask)
        self.label = ants.mask_image(self.label, self.BETmask)

        if target_shape is not None:
            # crop skull and air from image
            self.flair = ants.crop_image(self.flair, self.BETmask)
            self.dwi = ants.crop_image(self.dwi, self.BETmask)
            self.label = ants.crop_image(self.label, self.BETmask)

            # resample flair to desired shape
            self.flair = ants.resample_image(self.flair, target_spacing, use_voxels=False)
            self.flair = ants.pad_image(self.flair, target_shape)

            # resample other images to flair
            self.dwi = ants.resample_image_to_target(self.dwi, self.flair)
            self.label = ants.resample_image_to_target(self.label, self.flair, interpolation="genericLabel").astype("uint32")
            self.BETmask = ants.resample_image_to_target(self.BETmask, self.flair, interpolation="genericLabel").astype("uint32")

            # check shapes
            assert self.flair.shape == target_shape, f"Shape mismatch: FLAIR: {self.flair.shape}, target: {target_shape}"
            assert self.flair.spacing == target_spacing, f"Spacing mismatch: FLAIR: {self.flair.spacing}, target: {target_spacing}"
        assert self.flair.shape == self.dwi.shape == self.label.shape, f"Shape mismatch: FLAIR: {self.flair.shape}, DWI: {self.dwi.shape}, label: {self.label.shape}"
        assert self.flair.spacing == self.dwi.spacing == self.label.spacing, f"Spacing mismatch: FLAIR: {self.flair.spacing}, DWI: {self.dwi.spacing}, label: {self.label.spacing}"
        assert np.allclose(self.flair.direction, self.dwi.direction) and np.allclose(self.flair.direction, self.label.direction), f"Direction mismatch: FLAIR: {self.flair.direction}, DWI: {self.dwi.direction}, label: {self.label.direction}"
        assert (self.label.numpy() != 0).any(), "Label is empty"

    def free_data(self):
        """
        Frees the data by assigning paths to attributes: flair, dwi, label, and BETmask.
        """
        assert self.is_loaded(), f"Subject {self.name} is not loaded"

        self.flair = self._subj_paths[0]
        self.dwi = self._subj_paths[1]
        self.label = self._subj_paths[2]
        self.BETmask = self._subj_paths[3]

    def is_loaded(self):
        return isinstance(self.flair, ants.ANTsImage)

    def __post_init__(self):
        """
        Sets up transformation paths for FLAIR images.
        """
        flair_folder = os.path.dirname(self.flair)
        transform_flair_to_mni_folder = os.path.join(flair_folder, "flair_brain_to_mni")
        self.transform_flair_to_mni = [os.path.join(transform_flair_to_mni_folder, "warp.nii.gz"), os.path.join(transform_flair_to_mni_folder, "affine.mat")]
        self.transform_dwi_to_flair = os.path.join(flair_folder, "dwi_to_flair_affine.mat")

def ISLES2015(dataset_folder = "datasets/SISS2015_Training/") -> list[Subject]:
    """
    Generates a list of Subject objects for the ISLES 2015 dataset based on the provided dataset folder.
    
    Parameters:
    - dataset_folder: str, default is "datasets/SISS2015_Training/", the folder path containing the dataset
    
    Returns:
    - list[Subject]: a list of Subject objects, each representing a patient in the dataset with their associated FLAIR, DWI, and label paths
    """
    dataset_folder = dataset_folder
    subjects = []

    for patient_folder in range(1,29):
        scan_folders = os.listdir(f"datasets/SISS2015_Training/{patient_folder}")
        flair = next(filter(lambda x: "Flair" in x, scan_folders))
        dwi = next(filter(lambda x: "DWI" in x, scan_folders))
        label = next(filter(lambda x: "OT" in x, scan_folders))

        subjects.append(
            Subject(
                name = patient_folder,
                flair = f"{dataset_folder}/{patient_folder}/{flair}/{flair}.nii.gz",
                dwi = f"{dataset_folder}/{patient_folder}/{dwi}/{dwi}.nii.gz",
                label = f"{dataset_folder}/{patient_folder}/{label}/{label}.nii.gz"
            )
        )
    return subjects

def ISLES2022(dataset_folder = "datasets/ISLES-2022/") -> list[Subject]:
    """
    Generates a list of Subject objects for the ISLES 2022 dataset based on the provided dataset folder.
    
    Parameters:
    - dataset_folder: str, default is "datasets/ISLES-2022/", the folder path containing the dataset
    
    Returns:
    - list[Subject]: a list of Subject objects, each representing a patient in the dataset with their associated FLAIR, DWI, and label paths
    """
    subjects = []
    sub_strokecases = [f"sub-strokecase{i:04d}" for i in range(1,251)]
    for sub_strokecase in sub_strokecases:
        subjects.append(
            Subject(
                name = sub_strokecase,
                flair = f"{dataset_folder}/{sub_strokecase}/ses-0001/anat/{sub_strokecase}_ses-0001_FLAIR.nii.gz",
                dwi = f"{dataset_folder}/{sub_strokecase}/ses-0001/dwi/{sub_strokecase}_ses-0001_dwi.nii.gz",
                label = f"{dataset_folder}/derivatives/{sub_strokecase}/ses-0001/{sub_strokecase}_ses-0001_msk.nii.gz",
                labeled_modality = "dwi"
            )
        )
    return subjects

def Motol(dataset_folder = "datasets/Motol/") -> list[Subject]:
    """
    Generates a list of Subject objects for the Motol dataset based on the provided dataset folder.

    Parameters:
    - dataset_folder: str, default is "datasets/Motol/", the folder path containing the dataset

    Returns:
    - list[Subject]: a list of Subject objects, each representing a patient in the dataset with their associated FLAIR, DWI, label, and BETmask paths
    """
    subjects = []
    for patient_folder in os.listdir(dataset_folder):
        for anat in os.listdir(f'{dataset_folder}/{patient_folder}'):
            if anat == "Anat_20230109":
                # skip anat with corrupted segmentation
                # 2290867/Anat_20230109
                continue
            subjects.append(
                Subject(
                    name = f"{patient_folder}_{anat}",
                    flair = f"{dataset_folder}/{patient_folder}/{anat}/rFlair.nii.gz",
                    dwi = f"{dataset_folder}/{patient_folder}/{anat}/rDWI2.nii.gz",
                    label = f"{dataset_folder}/{patient_folder}/{anat}/Leze_FLAIR_DWI2.nrrd",
                    BETmask = f"{dataset_folder}/{patient_folder}/{anat}/BETmask.nii.gz"
                )
            )

    return subjects