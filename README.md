# Automated Segmentation of Brain Ischemic Stroke
[Project report in PDF](./doc_project/project.pdf)

# DeepMedic
For running DeepMedic, please preprocess the data using the provided script in `./datasets/` folder.

# Dataset
In this work we are using following datasets: Motol, ISLES 2022 and ISLES 2015. The dataset folders should be located in `./datasets/` and it contains unmodified data. For the ISLES 2022 and 2015 original folder structure is preserved. 

## Motol dataset
Motol dataset is provided by Second Faculty of Medicine CUNI, Prague.

In this work, we are using only FLAIR and DWI images and corresponding lesion masks thus only these files were included. Images in the Motol dataset are co-registered with SPM software.

In folder `./datasets/Motol` there are folders with the code of a particular patient. Because there are available scans in three different time points we have in each patient folder three other folders with names in the following format: `Anat_YYYYMMDD`. In each Anat folder there are three files:
- `rFlair.nii` - FLAIR image
- `rDWI2.nii` - DWI image
- `Leze_FLAIR_DWI2.nrrd` - manual lesion segmentation by the expert

After performing skull-stripping, each Anat folder contains also brain mask.

### Expected folder structure
```
./datasets/Motol/
│ 
└─── 98994
│   │
│   └─── Anat_20220413
│   │    │   Leze_FLAIR_DWI2.nrrd
│   │    │   rDWI2.nii
│   │    │   rFlair.nii
│   │
│   └─── Anat_20220420
│   │    │   ...
│   │
│   └─── Anat_20220425
│        │   ...
│
└─── ...
```