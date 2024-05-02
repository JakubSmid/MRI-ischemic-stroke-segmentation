#!/bin/bash

# set visible GPU
export CUDA_VISIBLE_DEVICES=1

DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# load modules
ml TensorFlow/2.0.0-fosscuda-2019b-Python-3.7.4

# activate venv
cd "$DIR/../venv_deepmedic/bin"
. "activate"

# run DeepMedic
cd "$DIR/../deepmedic"
./deepMedicRun -model "$DIR/model/modelConfig_vanilla.cfg"		        -train "$DIR/model/trainConfig_vanilla.cfg" -dev cuda
./deepMedicRun -model "$DIR/model/modelConfig_150FC.cfg"		        -train "$DIR/model/trainConfig_150FC.cfg" -dev cuda
./deepMedicRun -model "$DIR/model/modelConfig_ISLES2015.cfg" 		    -train "$DIR/model/trainConfig_ISLES2015.cfg" -dev cuda
./deepMedicRun -model "$DIR/model/modelConfig_image_augmentation.cfg" 	-train "$DIR/model/trainConfig_image_augmentation.cfg" -dev cuda
./deepMedicRun -model "$DIR/model/modelConfig_no_augmentation.cfg" 	    -train "$DIR/model/trainConfig_no_augmentation.cfg" -dev cuda
./deepMedicRun -model "$DIR/model/modelConfig_6layers.cfg" 		        -train "$DIR/model/trainConfig_6layers.cfg" -dev cuda
./deepMedicRun -model "$DIR/model/modelConfig_6layers_150FC.cfg" 	    -train "$DIR/model/trainConfig_6layers_150FC.cfg" -dev cuda
./deepMedicRun -model "$DIR/model/modelConfig_7layers.cfg" 		        -train "$DIR/model/trainConfig_7layers.cfg" -dev cuda
./deepMedicRun -model "$DIR/model/modelConfig_7layers_150FC.cfg" 	    -train "$DIR/model/trainConfig_7layers_150FC.cfg" -dev cuda
./deepMedicRun -model "$DIR/model/modelConfig_150FC_1L.cfg" 		    -train "$DIR/model/trainConfig_150FC_1L.cfg" -dev cuda
./deepMedicRun -model "$DIR/model/modelConfig_75FC.cfg" 		        -train "$DIR/model/trainConfig_75FC.cfg" -dev cuda

# train
#./deepMedicRun -model "$DIR/model/modelConfig_150FC_DWI.cfg"		-train "$DIR/model/trainConfig_150FC_DWI.cfg" -dev cuda
#./deepMedicRun -model "$DIR/model/modelConfig_150FC_FLAIR.cfg"		-train "$DIR/model/trainConfig_150FC_FLAIR.cfg" -dev cuda
