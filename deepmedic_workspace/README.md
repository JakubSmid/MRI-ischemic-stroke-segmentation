# DeepMedic
In this folder there is saved file `preprocessing.py` which loads the raw datasets and performs preprocessing which is recommended to be done before DeepMedic training. Preprocessed data are stored also in the new folder `./preprocessed/`.

There are also stored configuration files for each model and configuration files with paths to the training and test data. These files are stored in folders `./model` and `./train_ISLES_valid_Motol`.

There is provided bash script `run_deepmedic.sh` which can be used to run DeepMedic and train all configured models. After training, the trained models and its predictions are stored in `./output` folder.

For evaluation of models, there are following scripts:
- `evaluate_predictions_to_excel.py` - which creates excel file with evaluation for each case
- `parse_tensorboard.py` - which loads tensorboard logs, parses them and creates plots
- `plot_summary.py` - creates one plot with data from training from multiple models