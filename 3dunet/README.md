# 3D U-Net
This folder contains custom implementation of 3D U-Net and folder is used also as a workspace for this method.

Before training you need to run `preprocessing.py` which co-registers the data, reshapes them, applies brain mask and save them in `3dunet/raw` folder in the required structure.

Then you can run any training script which creates folder with logs and with trained weights. After that you can generate predictions using `predict.py`. Nothing special is needed.