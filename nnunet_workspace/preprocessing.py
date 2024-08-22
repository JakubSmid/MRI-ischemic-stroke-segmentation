import ants
import os
import multiprocessing

from datasets.utils import *
import datasets.dataset_loaders

def preprocessing(dataset: list[datasets.dataset_loaders.Subject],
                  output_folder: str = "nnunet_workspace/nnUNet_raw/"):
    """
    Preprocesses the dataset by creating necessary folders, loading data, and writing images using ANTs.

    Args:
        dataset (list[datasets.dataset_loaders.Subject]): The list of subjects to be preprocessed.
        output_folder (str): The path to the output folder where preprocessed data will be saved. 
            Defaults to "nnunet_workspace/nnUNet_raw/".

    Returns:
        None
    """
    os.makedirs(f"{output_folder}/Dataset001_Strokes/imagesTr", exist_ok=True)
    os.makedirs(f"{output_folder}/Dataset001_Strokes/labelsTr", exist_ok=True)
    
    N = len(dataset)
    for i, subj in enumerate(dataset):
        print(f"Processing {subj.name} ({i+1}/{N})...")

        subj.load_data()
        subj.extract_brain()
        subj.resample_to_target()
        subj.space_integrity_check()
        subj.empty_label_check()

        ants.image_write(subj.flair, f"{output_folder}/Dataset001_Strokes/imagesTr/{subj.name}_0000.nii.gz")
        ants.image_write(subj.dwi, f"{output_folder}/Dataset001_Strokes/imagesTr/{subj.name}_0001.nii.gz")
        ants.image_write(subj.label, f"{output_folder}/Dataset001_Strokes/labelsTr/{subj.name}.nii.gz")
        
        subj.free_data()

if __name__ == "__main__":
    # load datasets
    motol = datasets.dataset_loaders.Motol()
    isles2015 = datasets.dataset_loaders.ISLES2015()
    isles2022 = datasets.dataset_loaders.ISLES2022()

    # run preprocessing for each dataset in parallel
    motol_p = multiprocessing.Process(target=preprocessing,
                                      args=[motol])

    isles15_p = multiprocessing.Process(target=preprocessing,
                                        args=[isles2015])

    isles22_p = multiprocessing.Process(target=preprocessing,
                                        args=[isles2022])
    
    motol_p.start()
    isles15_p.start()
    isles22_p.start()