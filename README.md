# Automated Segmentation of Brain Ischemic Stroke
[Master's thesis PDF](./doc/Automaticka_segmentace_ischemicke_leze_u_cevni_mozkove_prihody.pdf)

## Repository structure
In this project, three deep learning stroke segmentation methods are tested on 3D MRI scans. Each of these methods is stored in its own folder in the repository. Configuration files and models of DeepMedic and nnU-Net are located in the folder `{Method Name}_workspace`.

The repository consists of four main parts.

The first part is located in the `datasets` folder. This folder should contain the individual datasets with the original folder structure. The folder also contains scripts for loading unmodified scans and scripts for working with them.

Before training the neural networks, the scans need to be coregistered by generating the appropriate transformations and skull stripping masks need to be generated. These steps are used to store the necessary files in the dataset folders. After this step, it is possible to do analyses of datasets using scripts in the `stats` folder and it is possible to start training neural networks.

The second main part is the `nnUNet` and `nnunet_workspace` folder, which contains the first of the tested methods. nnUNet is a method that uses 3D UNet with residual blocks as the backbone. The main advantage of nnUNet is the automatic data preprocessing and automatic neural network configuration and scaling of the UNet. The scripts for raw data conversion and nnUNet configuration are in the `nnunet_workspace` folder. Together with these files, the folder is used to store the preprocessed images and to store the results.

The third part is folder with `deepmedic` and `deepmedic_workspace`. These folders contains cloned DeepMedic repository and folder `deepmedic_workspace` contains the preprocessed data, configuration files and trained models.

The fourth part is the `3dunet` folder, which contains custom implementation of 3D U-Net. This folder also contains training scripts and trained models.

Summarized results of the experiments can be found in the `results` folder. There are csv files with statistics for each model, visualisations of the predictions and Jupyter notebook with analysis of the results.
