# Automatic MRI segmentation of ischemic stroke
[Project report in PDF](./doc_project/project.pdf)

## Notes
- [Data augmentation and preprocessing using TorchIO](https://www.imaios.com/en/resources/blog/ai-for-medical-imaging-using-torchio-python)

## Dataset
### ISLES-2022

### Motol
Dataset is provided by Second Faculty of Medicine CUNI, Prague.\
For this work we need FLAIR and DWI2 files in NIfTI format with .nii extension. Data should be already registered using SPM software.

Shape of each layer in dataset is (448, 448, 208). 

In folder `dataset` there are folders with code of particular patient. Because there are available scans in three different time points we have in each patient folder three other folders with name in following format: `Anat_YYYYMMDD`. In each Anat folder there are three files:
- `rFlair.nii` - FLAIR image
- `rDWI2.nii` - DWI2 image
- `Leze_FLAIR_DWI2.nrrd` - manual lesion segmentation by the expert

### Scheme of data structure
```
dataset   
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
