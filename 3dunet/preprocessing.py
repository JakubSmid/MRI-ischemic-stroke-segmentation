import ants
import os
import multiprocessing

from datasets.utils import *
import datasets.dataset_loaders as dataset_loaders

def preprocessing(dataset: list[dataset_loaders.Subject],
                  output_folder: str = "3dunet/raw/"):
    N = len(dataset)
    for i, subj in enumerate(dataset):
        # # skip empty subjects
        # if subj.name in ["sub-strokecase0150", "sub-strokecase0151", "sub-strokecase0170"]:
        #     continue

        # print progress
        print(f"Processing {i+1}/{N}: {subj.name}...")

        subj.load_data()
        subj.extract_brain()
        subj.resample_to_target()
        subj.space_integrity_check()
        subj.empty_label_check()

        # # count the number of voxels in the mask
        # bet_n_voxels = np.count_nonzero(subj.BETmask.numpy())
        # label_n_voxels = np.count_nonzero(subj.label.numpy())
        # bet_label_ratio = bet_n_voxels/label_n_voxels
        # subj.BETmask[subj.label] = round(bet_label_ratio)

        # prepare folder
        if not os.path.exists(f"{output_folder}/{subj.name}"):
            os.makedirs(f"{output_folder}/{subj.name}")

        # save images
        ants.image_write(subj.flair, f"{output_folder}/{subj.name}/flair.nii.gz")
        ants.image_write(subj.dwi, f"{output_folder}/{subj.name}/dwi.nii.gz")
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