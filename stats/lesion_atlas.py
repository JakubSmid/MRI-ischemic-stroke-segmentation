import ants
import numpy as np
import pandas as pd

import datasets.dataset_loaders as dataset_loaders

def generate_stat_lobes(dataset: list[dataset_loaders.Subject],
                        dataset_name: str,
                        template: ants.ants_image.ANTsImage,
                        atlas: ants.ants_image.ANTsImage, 
                        results_df: pd.DataFrame) -> None:
    # resample atlas to the template shape
    atlas = ants.resample_image_to_target(atlas, template, interpolation="genericLabel")
    atlas_np = atlas.numpy().astype(int)
    
    center_index = ants.transform_physical_point_to_index(atlas, [0, 0, 0])
    
    left_hemisphere_mask = np.zeros_like(atlas_np)
    left_hemisphere_mask[:round(center_index[0]), :, :] = 1

    right_hemisphere_mask = np.zeros_like(atlas_np)
    right_hemisphere_mask[round(center_index[0]):, :, :] = 1

    for i, subj in enumerate(dataset):
        print(f"Processing {i+1}/{len(dataset)}: {subj.name}...")
        subj.load_data()

        subj.extract_brain()
        subj.apply_transform_to_mni()
        subj.space_integrity_check()
        subj.empty_label_check()

        lesion_np = subj.label.numpy()

        for lobe in range(atlas_np.max()+1):
            lobe_mask = atlas_np == lobe
            left_lobe_mask = np.logical_and(lobe_mask, left_hemisphere_mask)
            right_lobe_mask = np.logical_and(lobe_mask, right_hemisphere_mask)

            left_lobe_lesion = lesion_np[left_lobe_mask].sum() * np.prod(subj.label.spacing) / 1000
            right_lobe_lesion = lesion_np[right_lobe_mask].sum() * np.prod(subj.label.spacing) / 1000

            results_df.loc[len(results_df)] = [dataset_name, subj.name, 'left', lobe, left_lobe_lesion]
            results_df.loc[len(results_df)] = [dataset_name, subj.name, 'right', lobe, right_lobe_lesion]

        subj.free_data()

if __name__ == "__main__":
    template = ants.image_read("datasets/template_flair_mni.nii.gz")
    atlas = ants.image_read("atlases/MNI Structural Atlas/MNI-maxprob-thr0-1mm.nii.gz")

    results_df = pd.DataFrame(columns=['Dataset', 'Subject', 'Hemisphere', 'Lobe', 'Volume [ml]'])

    dataset = dataset_loaders.ISLES2015()
    generate_stat_lobes(dataset, "ISLES2015", template, atlas, results_df)

    dataset = dataset_loaders.ISLES2022()
    generate_stat_lobes(dataset, "ISLES2022", template, atlas, results_df)

    dataset = dataset_loaders.Motol()
    generate_stat_lobes(dataset, "Motol", template, atlas, results_df)

    results_df.to_csv("results/stat_lobes.csv", index=False)

