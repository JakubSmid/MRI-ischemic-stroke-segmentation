# Stats

This folder contains scripts for statistical analysis of the datasets.

- `registration_similarity.py` - Calculates similarity between registered images. It is used only for checking registration quality and for verification of potential registration errors.
- `nibabel_ants_test.py` - Calculates timings for nibabel and ants processing of the datasets. It is used for comparing the performance of NiBabel and ANTs processing.
- `lesion_map.py` - Generates NIfTI image in MNI space for each dataset with sum of lesion masks. It allows to make quantitative comparisons between datasets.
- `lesion_map_img.py` - Generates images of "glass brain" from lesion maps created by `lesion_map.py`. Script projects maximum value of the lestion map to the MNI brain in frontal, axial and lateral directions.
- `lesion_map_stats.py` - Generates statistics of lesion occurrences in lobes using MNI Structural Atlas.
- `components_metadata.py` - Does component analysis and calculates shapes and sizes of images and labels. Also computes Dice coefficient after applying brain mask and resampling to the shape 200x200x200 (spacing 1x1x1).