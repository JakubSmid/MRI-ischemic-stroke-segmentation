import ants
import os
import multiprocessing

from datasets.utils import *
import datasets.dataset_loaders as dataset_loaders

def preprocessing(dataset: list[dataset_loaders.Subject],
                  output_folder: str = "deepmedic_workspace/raw/"):
    N = len(dataset)
    for i, subj in enumerate(dataset):
        # print progress
        print(f"Processing {i+1}/{N}: {subj.name}...")

        subj.load_data()
        subj.extract_brain()
        subj.resample_to_target()
        subj.space_integrity_check()
        subj.empty_label_check()

        # normalize scans within the mask
        n_flair = subj.flair.numpy()[subj.BETmask.numpy() == 1]
        n_flair = (subj.flair.numpy() - n_flair.mean()) / n_flair.std()
        n_dwi = subj.dwi.numpy()[subj.BETmask.numpy() == 1]
        n_dwi = (subj.dwi.numpy() - n_dwi.mean()) / n_dwi.std()
        flair = ants.new_image_like(subj.flair, n_flair)
        dwi = ants.new_image_like(subj.dwi, n_dwi)
        
        # prepare folder
        if not os.path.exists(f"{output_folder}/{subj.name}"):
            os.makedirs(f"{output_folder}/{subj.name}")

        # save images
        ants.image_write(flair, f"{output_folder}/{subj.name}/flair.nii.gz")
        ants.image_write(dwi, f"{output_folder}/{subj.name}/dwi.nii.gz")
        ants.image_write(subj.BETmask, f"{output_folder}/{subj.name}/BETmask.nii.gz")
        ants.image_write(subj.label, f"{output_folder}/{subj.name}/label.nii.gz")

        subj.free_data()

if __name__ == "__main__":
    # load datasets
    motol = dataset_loaders.Motol()
    isles2015 = dataset_loaders.ISLES2015()
    isles2022 = dataset_loaders.ISLES2022()

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