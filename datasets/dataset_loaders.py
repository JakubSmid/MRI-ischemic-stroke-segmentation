import os
import ants
import numpy as np
from dataclasses import dataclass
import datasets.utils

@dataclass
class Subject():
    name: str
    flair: str | ants.ants_image.ANTsImage
    dwi: str | ants.ants_image.ANTsImage
    label: str | ants.ants_image.ANTsImage
    labeled_modality: str = "flair"
    BETmask: str | ants.ants_image.ANTsImage = None

    def load_data(self, load_label=True, transform_to_flair=True):
        """
        Loads the subject data from file paths.

        Parameters:
            load_label (bool): Whether to load the label. Defaults to True.
            transform_to_flair (bool): Whether to transform the DWI to FLAIR space. Defaults to True.
        """
        assert not self.is_loaded(), f"Subject {self.name} is already loaded"

        # save paths
        self._subj_paths = [self.flair, self.dwi, self.label, self.BETmask]

        # load images
        self.flair = ants.image_read(self.flair)
        self.dwi = ants.image_read(self.dwi) 
        
        if transform_to_flair:
            transform = ants.read_transform(self.transform_dwi_to_flair)
            self.dwi = transform.apply_to_image(self.dwi, self.flair)

        if load_label:
            if ".nrrd" in self.label:
                label_flair, label_dwi = datasets.utils.load_nrrd(self.label)

                # resample flair label to flair
                if label_flair.shape != self.flair.shape:
                    label_flair = datasets.utils.resample_label_to_target(label_flair, self.flair)

                # resample dwi label to flair
                if label_dwi.shape != self.flair.shape:
                    label_dwi = datasets.utils.resample_label_to_target(label_dwi, self.flair)
                
                # apply transforms to label
                if transform_to_flair:
                    label_dwi = datasets.utils.apply_transform_to_label(label_dwi, transform, self.flair)
 
                label_union = np.logical_or(label_flair.numpy(), label_dwi.numpy()).astype(np.uint32)
                self.label = label_flair.new_image_like(label_union)

            else:
                self.label = ants.image_read(self.label).astype("uint32")
                if transform_to_flair and self.labeled_modality == "dwi":
                    self.label = datasets.utils.apply_transform_to_label(self.label, transform, self.flair)

        if self.BETmask:
            self.BETmask = ants.image_read(self.BETmask).astype("uint32")
        else:
            self.BETmask = self.flair.new_image_like((self.flair.numpy() != 0).astype("uint32"))

    def extract_brain(self):
        """
        Masks the FLAIR, DWI and label images of the subject by the brain mask.
        Results in the brain extracted FLAIR, DWI and label images.
        """
        assert self.is_loaded(), f"Subject {self.name} is not loaded"

        self.flair = ants.mask_image(self.flair, self.BETmask.astype("float32"))
        self.dwi = ants.mask_image(self.dwi, self.BETmask.astype("float32"))
        self.label = ants.mask_image(self.label, self.BETmask.astype("float32")).astype("uint32")

    def normalize(self):
        """
        Normalizes the FLAIR and DWI images of the subject by subtracting the mean and dividing by the standard deviation.
        """
        assert self.is_loaded(), f"Subject {self.name} is not loaded"

        data = self.flair.numpy()
        self.flair = self.flair.new_image_like((data-np.mean(data)) / np.std(data))

        data = self.dwi.numpy()
        self.dwi = self.dwi.new_image_like((data-np.mean(data)) / np.std(data))
    
    def resample_to_target(self, target_shape=(200, 200, 200), target_spacing=(1.0, 1.0, 1.0)):
        """
        Resamples the subject to the target shape and spacing.

        Parameters:
            target_shape (tuple[int, int, int], optional): Desired shape of the output image. Defaults to (200, 200, 200).
            target_spacing (tuple[float, float, float], optional): Desired spacing of the output image. Defaults to (1.0, 1.0, 1.0).
        """
        assert self.is_loaded(), f"Subject {self.name} is not loaded"

        # crop skull and air from image
        self.flair = ants.crop_image(self.flair, self.BETmask)
        self.dwi = ants.crop_image(self.dwi, self.BETmask)
        self.label = ants.crop_image(self.label, self.BETmask)

        # resample flair to desired shape
        self.flair = ants.resample_image(self.flair, target_spacing, use_voxels=False)
        self.flair = ants.pad_image(self.flair, target_shape)

        # resample other images to flair
        self.dwi = ants.resample_image_to_target(self.dwi, self.flair)
        self.label = datasets.utils.resample_label_to_target(self.label, self.flair)
        self.BETmask = datasets.utils.resample_label_to_target(self.BETmask, self.flair)

        # check shapes
        assert self.flair.shape == target_shape, f"Shape mismatch: FLAIR: {self.flair.shape}, target: {target_shape}"
        assert self.flair.spacing == target_spacing, f"Spacing mismatch: FLAIR: {self.flair.spacing}, target: {target_spacing}"

    def apply_transform_to_mni(self, template_mni="datasets/template_flair_mni.nii.gz"):
        """
        Applies the transformation from the FLAIR space of the subject to the MNI template space.
        Transformation is applied to the FLAIR, DWI, label and BET images.

        The transformation is computed from the displacement field and affine transform files.
        
        Parameters:
            template_mni (str): The path to the MNI template image. Default is "datasets/template_flair_mni.nii.gz".
        """
        assert self.is_loaded(), f"Subject {self.name} is not loaded"

        template_mni = ants.image_read(template_mni)
        warp = ants.transform_from_displacement_field(ants.image_read(self.transform_flair_to_mni[0]))
        affine = ants.read_transform(self.transform_flair_to_mni[1])

        self.flair = affine.apply_to_image(self.flair, template_mni)
        self.flair = warp.apply_to_image(self.flair, template_mni)

        self.dwi = affine.apply_to_image(self.dwi, template_mni)
        self.dwi = warp.apply_to_image(self.dwi, template_mni)

        self.BETmask = datasets.utils.apply_transform_to_label(self.BETmask, affine, template_mni)
        self.BETmask = datasets.utils.apply_transform_to_label(self.BETmask, warp, template_mni)

        self.label = datasets.utils.apply_transform_to_label(self.label, affine, template_mni)
        self.label = datasets.utils.apply_transform_to_label(self.label, warp, template_mni)

    def space_integrity_check(self):
        """
        Checks the spatial integrity of a subject by ensuring that the FLAIR, DWI, and label images
        have the same shape and spacing, and that the direction of the images is the same.

        Raises:
            AssertionError: if the shape, spacing, or direction of the images do not match
        """
        assert self.is_loaded(), f"Subject {self.name} is not loaded"

        assert self.flair.shape == self.dwi.shape == self.label.shape, f"Shape mismatch: FLAIR: {self.flair.shape}, DWI: {self.dwi.shape}, label: {self.label.shape}"
        assert self.flair.spacing == self.dwi.spacing == self.label.spacing, f"Spacing mismatch: FLAIR: {self.flair.spacing}, DWI: {self.dwi.spacing}, label: {self.label.spacing}"
        assert np.allclose(self.flair.direction, self.dwi.direction) and np.allclose(self.flair.direction, self.label.direction), f"Direction mismatch: FLAIR: {self.flair.direction}, DWI: {self.dwi.direction}, label: {self.label.direction}"

    def empty_label_check(self):
        """
        Checks if the label is empty. If it is, raises an assertion error.
        
        For subject sub-strokecase0150, sub-strokecase0151, sub-strokecase0170 empty labels are allowed, beacause there is no stroke in these cases.

        Raises:
            AssertionError: if the label is empty
        """
        assert self.is_loaded(), f"Subject {self.name} is not loaded"
        assert (self.label.numpy() != 0).any() or ("sub-strokecase0150" in self.name) or ("sub-strokecase0151" in self.name) or ("sub-strokecase0170" in self.name), f"Subject {self.name} label is empty"

    def save(self, output_folder):
        """
        Saves the subject's data to the given output folder.

        Parameters:
            output_folder (str): The folder where the data should be saved.
        """
        assert self.is_loaded(), f"Subject {self.name} is not loaded"

        ants.image_write(self.flair, os.path.join(output_folder, f"{self.name}_flair.nii.gz"))
        ants.image_write(self.dwi, os.path.join(output_folder, f"{self.name}_dwi.nii.gz"))
        ants.image_write(self.label, os.path.join(output_folder, f"{self.name}_label.nii.gz"))
        ants.image_write(self.BETmask, os.path.join(output_folder, f"{self.name}_BETmask.nii.gz"))

    def free_data(self):
        """
        Frees the data by reassigning paths to attributes: flair, dwi, label, and BETmask.
        """
        assert self.is_loaded(), f"Subject {self.name} is not loaded"

        self.flair = self._subj_paths[0]
        self.dwi = self._subj_paths[1]
        self.label = self._subj_paths[2]
        self.BETmask = self._subj_paths[3]

    def is_loaded(self) -> bool:
        """
        Checks if the subject data is loaded. I.e. if the flair attribute is an ANTs image.
        
        Returns:
            bool: True if the subject is loaded, False otherwise.
        """
        return isinstance(self.flair, ants.ants_image.ANTsImage)

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
        dataset_folder: str, default is "datasets/SISS2015_Training/", the folder path containing the dataset
    
    Returns:
        list (Subject): a list of Subject objects, each representing a patient in the dataset with their associated FLAIR, DWI, and label paths
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
                name = str(patient_folder),
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
        dataset_folder: str, default is "datasets/ISLES-2022/", the folder path containing the dataset
    
    Returns:
        list (Subject): a list of Subject objects, each representing a patient in the dataset with their associated FLAIR, DWI, and label paths
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
        dataset_folder: str, default is "datasets/Motol/", the folder path containing the dataset

    Returns:
        list (Subject): a list of Subject objects, each representing a patient in the dataset with their associated FLAIR, DWI, label, and BETmask paths
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