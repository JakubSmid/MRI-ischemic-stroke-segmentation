# nnUNet
In this folder there are scripts used for loading modules on HPC, scripts for converting raw datasets to the nnUNet format and configuration files of the nnUNet.

First of all, you need to run `preprocessing.py` which co-registers the data, reshapes them, applies brain mask and save them in `nnunet_workspace/nnUNet_raw` folder. If you want to use MNI space, then run `preprocessing_mni.py` instead. After dataset conversion to nnUNet format, there will be a new folder `nnUNet_raw` with corresponding dataset folder and its files. Now you should copy `dataset.json` into `nnUNet_raw/Datasetxxx_DatasetName/`, which is [configuration file for nnUNet](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md#datasetjson).

Before running nnUNet preprocessing, please source `load_nnunet.sh` to set up the environment and [install nnUNet](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md). Now source `load_nnunet.sh` again and start nnUNet preprocessing `nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity -c 3d_fullres -pl nnUNetPlannerResEncM`.

This will create `nnunet_workspace/nnUNet_preprocessed/Datasetxxx_DatasetName` folder with the data, fingerprint and UNet configuration. Configuration of the neural network is saved in `nnUNetPlannerResEncM.json` and it is possible to [modify it](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/explanation_plans_files.md). nnUNet uses 5-fold cross-validation by default, but we want to use ISLES 2022 and ISLES 2015 as train set and Motol as test set. Thus we should rewrite default splits. You need to copy `splits_final_ISLES_train.json` into `nnunet_workspace/nnUNet_preprocessed/Datasetxxx_DatasetName/splits_final.json` folder.

Use following command to train on nnUNet `nnUNetv2_train DATASET_ID 3d_fullres 0 -p nnUNetResEncUNetMPlans`. Trainer is edited to use 500 epochs instead of 1000. Trained model with debug info, UNet configuration and predictions are saved in `nnunet_workspace/nnUNet_results/Datasetxxx_DatasetName`.

Predictions can be generated using following command: `nnUNetv2_predict -i input_folder -o output_folder -d DATASET_ID -c 3d_fullres -p nnUNetResEncUNetMPlans -f 0 --save_probabilities`. This will run inference using the last checkpoint saved from training. Predictions will be saved with its probability maps, which allows to use output fusion. Beware that in input folder there must be preserved numbering of the images i. e. image_0000.nii.gz for FLAIR and image_0001.nii.gz for DWI. And scans must be transformed same as during training (using `preprocessing.py` or `preprocessing_mni.py`).

Ensembling can be done simply with `nnUNetv2_ensemble -i folder1 folder2 ... -o output_folder`.

## Motol ensemble
To achieve the best possible results on Motol dataset, there is also provided `splits_final_Motol_ensemble.json`, which can be used for 5-fold cross-validation on Motol dataset.

Because Motol dataset contains three scans of the same patient in different times, there are manually created splits which ensure that all three scans of the same patient are either in train or validation set. For nnUNet preprocessing and training it can be used images genereated by `preprocessing.py` or `preprocessing_mni.py`.