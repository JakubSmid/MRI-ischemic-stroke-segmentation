import ants
import shutil
import multiprocessing
import os
import datasets.dataset_loaders as dataset_loaders

def registration_SyN(fixed: ants.ants_image.ANTsImage, moving: ants.ants_image.ANTsImage, output_files: list[str]):
    """
    Perform SyN registration between two ANTs images and save the resulting transforms.

    Parameters:
        fixed (ants.ants_image.ANTsImage): The fixed image for registration.
        moving (ants.ants_image.ANTsImage): The moving image for registration.
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

def registration_Rigid(fixed: ants.ants_image.ANTsImage, moving: ants.ants_image.ANTsImage, output_file: str):
    """
    Perform Rigid registration between two ANTs images and save the resulting transform.

    Parameters:
        fixed (ants.ants_image.ANTsImage): The fixed image for registration.
        moving (ants.ants_image.ANTsImage): The moving image for registration.
        output_file (str): The output file path to save the transform.

    Returns:
        None
    """
    # apply registration
    mytx = ants.registration(fixed=fixed, moving=moving, type_of_transform="Rigid")

    # save transform
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    shutil.move(mytx['fwdtransforms'][0], output_file)

def registration(dataset: list[dataset_loaders.Subject], template_mni: ants.ants_image.ANTsImage):
    """
    Perform registration of the dataset for each subject using SyN and Affine transformations.
    
    Parameters:
        dataset (list[dataset_loaders.Subject]): List of subjects with MRI data.
        template_mni (ants.ants_image.ANTsImage): The template image for registration to MNI space.
    
    Returns:
        None
    """
    for i, subj in enumerate(dataset):
        print(f"Processing {subj.name} ({i+1}/{len(dataset)})...")

        subj.load_data(load_label=False, transform_to_flair=False)
        flair_masked = ants.mask_image(subj.flair, subj.BETmask.astype("float32"))

        registration_SyN(template_mni, flair_masked, subj.transform_flair_to_mni)
        registration_Rigid(subj.flair, subj.dwi, subj.transform_dwi_to_flair)

        subj.free_data()

if __name__ == "__main__":
    template_mni = ants.image_read("datasets/template_flair_mni.nii.gz")

    dataset = dataset_loaders.Motol()
    p_Motol = multiprocessing.Process(target=registration, args=(dataset, template_mni))

    dataset = dataset_loaders.ISLES2015()
    p_ISLES15 = multiprocessing.Process(target=registration, args=(dataset, template_mni))

    dataset = dataset_loaders.ISLES2022()
    p_ISLES22 = multiprocessing.Process(target=registration, args=(dataset, template_mni))

    p_Motol.start()
    p_ISLES15.start()
    p_ISLES22.start()

    p_Motol.join()
    p_ISLES15.join()
    p_ISLES22.join()
