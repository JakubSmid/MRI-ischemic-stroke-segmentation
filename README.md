# Automated Segmentation of Brain Ischemic Stroke
[Project report in PDF](./doc_project/project.pdf)

## Repository structure
In this project, three methods using deep learning for stroke segmentation are tested on 3D MRI scans. Each of these methods is stored in its own folder in the repository and configuration files and models are located in the folder `{Method Name}_workspace`.

The repository consists of four main parts.

The first part is located in the `datasets` folder. This folder should contain the individual datasets with the original folder structure. The folder also contains scripts for loading unmodified scans and scripts for working with them.

Before training the neural networks, the scans need to be coregistered by generating the appropriate transformations and skull stripping masks need to be generated. These steps are used to store the necessary files in the dataset folders. After this step, it is possible to do analyses of the datasets using the scripts in the `stats` folder and start training the neural networks.

The second main part is the `nnUNet` and `nnunet_workspace` folder, which contains the first of the tested methods. nnUNet is a method that uses 3D UNet with residual blocks as the backbone. The main adventage of nnUNet is the automatic data preprocessing and automatic neural network configuration and scaling of the UNet. The scripts for raw data conversion and nnUNet configuration are in the `nnunet_workspace` folder. Together with these files, the folder is used to store the preprocessed images and to store the results.