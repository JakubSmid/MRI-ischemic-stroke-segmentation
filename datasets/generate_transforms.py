import ants
import shutil
import multiprocessing

import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import datasets.dataset_loaders

def registration_SyN(fixed: ants.ANTsImage, moving: ants.ANTsImage, output_files: list[str]):
    """
    Perform SyN registration between two ANTs images and save the resulting transforms.

    Parameters:
        fixed (ants.ANTsImage): The fixed image for registration.
        moving (ants.ANTsImage): The moving image for registration.
        output_files (list[str]): List of output file paths to save the transforms.

    Returns:
        None
    """
    # apply registration
    mytx = ants.registration(fixed=fixed, moving=moving, type_of_transform="SyN")

    # save transform
    os.makedirs(os.path.dirname(output_files[0]), exist_ok=True)
    shutil.move(mytx['fwdtransforms'][0], output_files[0])
    shutil.move(mytx['fwdtransforms'][1], output_files[1])

def registration_Affine(fixed: ants.ANTsImage, moving: ants.ANTsImage, output_file: str):
    """
    Perform Affine registration between two ANTs images and save the resulting transform.

    Parameters:
        fixed (ants.ANTsImage): The fixed image for registration.
        moving (ants.ANTsImage): The moving image for registration.
        output_file (str): The output file path to save the transform.

    Returns:
        None
    """
    # apply registration
    mytx = ants.registration(fixed=fixed, moving=moving, type_of_transform="Affine")

    # save transform
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    shutil.move(mytx['fwdtransforms'][0], output_file)

def registration_Motol(dataset: list[datasets.dataset_loaders.Subject], template_mni: ants.ANTsImage):
    """
    Perform registration of the MOTOL dataset for each subject in the dataset using SyN and Affine transformations.
    
    Parameters:
        dataset (list[datasets.dataset_loaders.Subject]): List of subjects with MRI data.
        template_mni (ants.ANTsImage): The template image for registration to MNI space.
    
    Returns:
        None
    """
    for i, subj in enumerate(dataset):
        print(f"Processing {subj.name} ({i+1}/{len(dataset)})...")

        # load data
        flair = ants.image_read(subj.flair)
        dwi = ants.image_read(subj.dwi)
        bet = ants.image_read(subj.BETmask)

        # apply BET
        flair_masked = ants.mask_image(flair, bet)

        registration_SyN(template_mni, flair_masked, subj.transform_flair_to_mni)
        registration_Affine(flair, dwi, subj.transform_dwi_to_flair)

def registration_ISLES(dataset: list[datasets.dataset_loaders.Subject], template_mni: ants.ANTsImage):
    """
    Perform registration for each subject in the dataset using SyN and Affine transformations.
    
    Parameters:
        dataset (list[datasets.dataset_loaders.Subject]): List of subjects with MRI data.
        template_mni (ants.ANTsImage): The template image for registration to MNI space.
    
    Returns:
        None
    """
    for i, subj in enumerate(dataset):
        print(f"Processing {subj.name} ({i+1}/{len(dataset)})...")

        # load data
        dwi = ants.image_read(subj.dwi)
        flair = ants.image_read(subj.flair)

        registration_SyN(template_mni, flair, subj.transform_flair_to_mni)
        registration_Affine(flair, dwi, subj.transform_dwi_to_flair)

if __name__ == "__main__":
    template_mni = ants.image_read("datasets/template_flair_mni.nii.gz")

    dataset = datasets.dataset_loaders.Motol()
    p_Motol = multiprocessing.Process(target=registration_Motol, args=(dataset, template_mni))

    dataset = datasets.dataset_loaders.ISLES2015()
    p_ISLES15 = multiprocessing.Process(target=registration_ISLES, args=(dataset, template_mni))

    dataset = datasets.dataset_loaders.ISLES2022()
    p_ISLES22 = multiprocessing.Process(target=registration_ISLES, args=(dataset, template_mni))

    p_Motol.start()
    p_ISLES15.start()
    p_ISLES22.start()

    p_Motol.join()
    p_ISLES15.join()
    p_ISLES22.join()
