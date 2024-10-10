# DeepMedic
In this folder there are scripts used for loading modules on HPC, scripts for converting raw datasets to the DeepMedic format and configuration files of the DeepMedic.

First of all, you need to run `preprocessing.py` which co-registers the data, reshapes them, applies brain mask and save them in `deepmedic_workspace/raw` folder in the required structure. After dataset conversion to DeepMedic format, there will be a new folder `raw` with corresponding dataset folder and its files.

Before running DeepMedic preprocessing, please source `source ./venv_deepmedic/bin/activate` to set up the environment and [install DeepMedic](https://github.com/deepmedic/deepmedic/blob/master/documentation/README.md#12-installation).

Folder `deepmedic_workspace/train_ISLES_valid_Motol` contains configuration files with paths to the training and validation data. The main folder is `deepmedic_workspace/model` where are stored configuration files for each model.

You can train all models by running `run_deepmedic.sh` or you can look at `run_deepmedic.sh` and comment out some models if you want to train only some.

After training, the trained models and its predictions are stored in `deepmedic_workspace/output` folder. Output folder includes logs and tensorboard plots. It is possible to export these plots to image using `parse_tensorboard.py` or `plot_summary.py`.

To make predictions you need to have original configuration files in `deepmedic_workspace/model`. To run inference you can modify configs in `./test/` and run the following command:
```
./deepmedic/deepMedicRun -model deepmedic_workspace/model/modelConfig*.cfg -test deepmedic_workspace/test/config.cfg -load deepmedic_workspace/output/saved_models/*/*.final.*.model.ckpt -dev cuda
```