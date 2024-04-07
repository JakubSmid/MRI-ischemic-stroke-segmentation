# nnUNet
First you need to run `preprocessing.py` which co-registers the data, reshapes them, applies brain mask and save them in `nnunet_workspace/nnUNet_raw` folder. Then please source `load_nnunet.sh` to set up the environment.

`preprocessing.py` doesn't create `dataset.json` file, so you should use generate_dataset_json function from `nnunetv2/dataset_conversion/generate_dataset_json.py`.

When folder nnUNet_raw is ready, you can start nnUNet preprocessing `nnUNetv2_plan_and_preprocess -c 3d_fullres -d DATASET_ID --verify_dataset_integrity`. After this step, splits_final.json and plans.json are generated inside `nnunet_workspace/nnUNet_preprocessed`. You can modify them if needed - see [manual data splits](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/manual_data_splits.md) and [modifying the nnU-Net Configurations](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/explanation_plans_files.md).

