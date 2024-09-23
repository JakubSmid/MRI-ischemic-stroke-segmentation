#!/bin/bash

# set visible GPU
export CUDA_VISIBLE_DEVICES=2

# deactivate old venv
deactivate
ml purge

# activate venv
ml TensorFlow/2.6.0-foss-2021a-CUDA-11.3.1
. "./venv_deepmedic/bin/activate"

# run DeepMedic
./deepmedic/deepMedicRun -model "deepmedic_workspace/model/modelConfig_vanilla.cfg"             -train "deepmedic_workspace/model/trainConfig_vanilla.cfg" -dev cuda
./deepmedic/deepMedicRun -model "deepmedic_workspace/model/modelConfig_150FC.cfg"		        -train "deepmedic_workspace/model/trainConfig_150FC.cfg" -dev cuda
./deepmedic/deepMedicRun -model "deepmedic_workspace/model/modelConfig_ISLES2015.cfg" 		    -train "deepmedic_workspace/model/trainConfig_ISLES2015.cfg" -dev cuda
./deepmedic/deepMedicRun -model "deepmedic_workspace/model/modelConfig_image_augmentation.cfg"  -train "deepmedic_workspace/model/trainConfig_image_augmentation.cfg" -dev cuda
./deepmedic/deepMedicRun -model "deepmedic_workspace/model/modelConfig_no_augmentation.cfg" 	-train "deepmedic_workspace/model/trainConfig_no_augmentation.cfg" -dev cuda
./deepmedic/deepMedicRun -model "deepmedic_workspace/model/modelConfig_6layers.cfg" 		    -train "deepmedic_workspace/model/trainConfig_6layers.cfg" -dev cuda
./deepmedic/deepMedicRun -model "deepmedic_workspace/model/modelConfig_6layers_150FC.cfg" 	    -train "deepmedic_workspace/model/trainConfig_6layers_150FC.cfg" -dev cuda
./deepmedic/deepMedicRun -model "deepmedic_workspace/model/modelConfig_7layers.cfg" 		    -train "deepmedic_workspace/model/trainConfig_7layers.cfg" -dev cuda
./deepmedic/deepMedicRun -model "deepmedic_workspace/model/modelConfig_7layers_150FC.cfg" 	    -train "deepmedic_workspace/model/trainConfig_7layers_150FC.cfg" -dev cuda
./deepmedic/deepMedicRun -model "deepmedic_workspace/model/modelConfig_150FC_1L.cfg" 		    -train "deepmedic_workspace/model/trainConfig_150FC_1L.cfg" -dev cuda
./deepmedic/deepMedicRun -model "deepmedic_workspace/model/modelConfig_75FC.cfg" 		        -train "deepmedic_workspace/model/trainConfig_75FC.cfg" -dev cuda

# train
#./deepmedic/deepMedicRun -model "deepmedic_workspace/model/modelConfig_75FC_DWI.cfg"		-train "deepmedic_workspace/model/trainConfig_75FC_DWI.cfg" -dev cuda
#./deepmedic/deepMedicRun -model "deepmedic_workspace/model/modelConfig_75FC_FLAIR.cfg"		-train "deepmedic_workspace/model/trainConfig_75FC_FLAIR.cfg" -dev cuda
