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
./deepMedicRun -model "$DIR/debug_model/modelConfig_debug1.cfg"	-train "$DIR/debug_model/trainConfig_debug1.cfg" -dev cuda
./deepMedicRun -model "$DIR/debug_model/modelConfig_debug2.cfg"	-train "$DIR/debug_model/trainConfig_debug2.cfg" -dev cuda
