import ants
import cc3d
import numpy as np
import pandas as pd
import datasets.dataset_loaders as dataset_loaders
import datasets.utils as utils

def components(subject: dataset_loaders.Subject, connectivity=26) -> pd.DataFrame:
    """
    Compute connected components of a label image and compute volume of each component.

    Parameters:
        subject (dataset_loaders.Subject): The subject to process.
        connectivity (int, optional): The connectivity to use for the connected components. Defaults to 26.

    Returns:
        pd.DataFrame: A DataFrame with columns 'name' and 'volume_ml', where each row corresponds to a component.
    """
    df_components = pd.DataFrame()
    components = cc3d.connected_components(subject.label.numpy(), connectivity=26)
    for component in np.unique(components)[1:]:
        volume = utils.voxel_count_to_volume_ml(np.count_nonzero(components == component), subject.label.spacing)
        stats = {
            "name": subject.name,
            "volume_ml": volume
        }
        df_components = pd.concat([df_components, pd.DataFrame([stats])])
    return df_components

def stats_Motol(dataset: list[dataset_loaders.Subject],
                dataset_name: str):
    """
    Compute statistics for the Motol dataset.

    Parameters:
        dataset (list[dataset_loaders.Subject]): List of subjects with MRI data.
        dataset_name (str): Name of the dataset.
    """
    df_cases = pd.DataFrame()
    df_components = pd.DataFrame()

    for i, subj in enumerate(dataset):
        print(f"Processing {i+1}/{len(dataset)}: {subj.name}...")
        dwi = ants.image_read(subj.dwi)
        flair = ants.image_read(subj.flair)
        label_flair, label_dwi = utils.load_nrrd(subj.label)
        
        subj.load_data()
        
        # compute components
        df_components = pd.concat([df_components, components(subj)])

        label_volume = utils.voxel_count_to_volume_ml(np.count_nonzero(subj.label.numpy()), subj.label.spacing)
        label_flair_volume = utils.voxel_count_to_volume_ml(np.count_nonzero(label_flair.numpy()), label_flair.spacing)
        label_dwi_volume = utils.voxel_count_to_volume_ml(np.count_nonzero(label_dwi.numpy()), label_dwi.spacing)
        bet_mask_volume_ml = utils.voxel_count_to_volume_ml(np.count_nonzero(subj.BETmask.numpy()), subj.BETmask.spacing)
        stats = {
            "name": subj.name,
            "shape_flair": flair.shape,
            "shape_dwi": dwi.shape,
            "voxel_dim_flair": flair.spacing,
            "voxel_dim_dwi": dwi.spacing,
            "lesion_volume_ml": label_volume,
            "flair_lesion_volume_ml": label_flair_volume,
            "dwi_lesion_volume_ml": label_dwi_volume,
            "bet_mask_volume_ml": bet_mask_volume_ml
        }

        label_before_preprocessing = subj.label
        subj.extract_brain()
        subj.resample_to_target()

        # compare label after brain extraction and resampling to 200x200x200, 1x1x1
        processed_img_resampled = utils.resample_label_to_target(subj.label, label_before_preprocessing.astype("float32"))
        stats["dice_after_preprocessing"] = utils.dice_coefficient(label_before_preprocessing.numpy(), processed_img_resampled.numpy())

        df_cases = pd.concat([df_cases, pd.DataFrame([stats])])
    
    df_cases.to_csv(f"results/{dataset_name}_stats.csv")
    df_components.to_csv(f"results/{dataset_name}_components.csv")

def stats_ISLES(dataset: list[dataset_loaders.Subject],
                dataset_name: str):
    """
    Compute statistics for the ISLES dataset.

    Parameters:
        dataset (list[dataset_loaders.Subject]): List of subjects with MRI data.
        dataset_name (str): Name of the dataset.
    """
    df_cases = pd.DataFrame()
    df_components = pd.DataFrame()

    for i, subj in enumerate(dataset):
        print(f"Processing {i+1}/{len(dataset)}: {subj.name}...")
        dwi = ants.image_read(subj.dwi)
        flair = ants.image_read(subj.flair)
        label = ants.image_read(subj.label).astype("uint32")
        subj.load_data()

        # compute components
        df_components = pd.concat([df_components, components(subj)])

        # compute volumes
        label_volume = utils.voxel_count_to_volume_ml(np.count_nonzero(label.numpy()), label.spacing)
        bet_mask_volume = utils.voxel_count_to_volume_ml(np.count_nonzero(subj.BETmask.numpy()), subj.BETmask.spacing)
        stats = {
            "name": subj.name,
            "shape_flair": flair.shape,
            "shape_dwi": dwi.shape,
            "voxel_dim_flair": flair.spacing,
            "voxel_dim_dwi": dwi.spacing,
            "lesion_volume_ml": label_volume,
            "bet_mask_volume_ml": bet_mask_volume
        }

        label_before_preprocessing = subj.label
        subj.extract_brain()
        subj.resample_to_target()

        # compare label after brain extraction and resampling to 200x200x200, 1x1x1
        processed_img_resampled = utils.resample_label_to_target(subj.label, label_before_preprocessing.astype("float32"))
        stats["dice_after_preprocessing"] = utils.dice_coefficient(label_before_preprocessing.numpy(), processed_img_resampled.numpy())

        df_cases = pd.concat([df_cases, pd.DataFrame([stats])])

        subj.free_data()

    # save dataframes
    df_cases.to_csv(f"results/{dataset_name}_metadata.csv", index=False)
    df_components.to_csv(f"results/{dataset_name}_components.csv", index=False)

if __name__ == "__main__":
    stats_Motol(dataset_loaders.Motol(), "Motol")
    stats_ISLES(dataset_loaders.ISLES2015(), "ISLES2015")
    stats_ISLES(dataset_loaders.ISLES2022(), "ISLES2022")
