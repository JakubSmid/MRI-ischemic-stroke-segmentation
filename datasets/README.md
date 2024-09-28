# Datasets
This folder contains three folders with unmodified datasets (Motol, SISS2015_Training and ISLES-2022) with preserved folder and file structure.

In this folder there are three scripts:
- `download_Motol.py` - Executable script which has been used to download Motol dataset from the NAS drive to the local machine.
- `dataset_loaders.py` - Contains definition of Subject class which is used for loading images and spatial transformations. Subject class is universal for all datasets. File also contains functions for loading each dataset as a list of Subjects.
- `generate_transforms.py` - Contains functions for registration of brain MRI scans using ANTs. There are two types of registration: Rigid and SyN. Rigid registration is used for transformation from DWI to FLAIR space. SyN registration is used for transformation from FLAIR to MNI space. Transformation files are saved in each subject folder.
- `utils.py` - Contains utility functions which are used mainly for preprocessing.

## Motol
Motol dataset is provided by Second Faculty of Medicine CUNI, Prague.

In this work, I am using only FLAIR and DWI images and corresponding lesion masks labeled by the expert. Images in the Motol dataset are co-registered with SPM software.

In folder `./Motol` there are folders with the code of a particular patient. Because there are available scans in three different time points we have in each patient folder three other folders with names in the following format: `Anat_YYYYMMDD`. In each Anat folder there are three files:
- `rFlair.nii.gz` - FLAIR image
- `rDWI2.nii.gz` - DWI image
- `Leze_FLAIR_DWI2.nrrd` - manual lesion segmentation by the expert

Scan number `2290867/Anat_20230109` has corrupted lesion segmentation file, thus it is excluded from dataset loading.

### Folder structure
```
Motol
├── 115346
│   ├── Anat_20220413
│   │   ├── Leze_FLAIR_DWI2.nrrd
│   │   ├── rDWI2.nii.gz
│   │   └── rFlair.nii.gz
│   ├── Anat_20220420
│   │   ├── Leze_FLAIR_DWI2.nrrd
│   │   ├── rDWI2.nii.gz
│   │   └── rFlair.nii.gz
│   └── Anat_20220425
│       ├── Leze_FLAIR_DWI2.nrrd
│       ├── rDWI2.nii.gz
│       └── rFlair.nii.gz
├── ...
```

## ISLES 2015
To extend Motol dataset I am also using ISLES-2015 dataset which can be downloaded from [https://www.isles-challenge.org/ISLES2015/](https://www.isles-challenge.org/ISLES2015/). Because Motol dataset contains sub-acute stokes I'm using only data from SISS (sub-acute ischemic stroke lesion segmentation) task of ISLES-2015 which have publicly available ground truth segmentations (training cases).

<cite>Maier, Oskar et al. “ISLES 2015 - A public evaluation benchmark for ischemic stroke lesion segmentation from multispectral MRI.” Medical image analysis vol. 35 (2017): 250-269. doi:10.1016/j.media.2016.07.009</cite>

## ISLES 2022
From ISLES 2022 I'm using data from Multimodal MRI infarct segmentation in acute and sub-acute stroke task which can be downloaded from [https://isles22.grand-challenge.org/](https://isles22.grand-challenge.org/).

<cite>Hernandez Petzsche, Moritz R et al. “ISLES 2022: A multi-center magnetic resonance imaging stroke lesion segmentation dataset.” Scientific data vol. 9,1 762. 10 Dec. 2022, doi:10.1038/s41597-022-01875-5</cite>