import os
import logging
import time
import argparse

import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adamax
from torch.utils.tensorboard import SummaryWriter

from torchmetrics.classification import Dice

import torchio as tio

from model import UNet3D
from dataset import load_dataset


parser = argparse.ArgumentParser()
parser.add_argument("model_folder", type=str)
parser.add_argument("--channel", type=str)
parser.add_argument("--probabilities", action="store_true")
parser.add_argument("--epoch", type=int, default=20)
args = parser.parse_args()

os.makedirs(f"{args.model_folder}/predictions", exist_ok=True)

# set up logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s [%(name)s] %(message)s",
    datefmt="%d/%b/%Y %H:%M:%S"
    )

logger = logging.getLogger(__name__)

preprocessing_transform = tio.Compose([
    tio.Pad((48, 48, 48)),
    tio.ZNormalization()
])

# load dataloaders and model
valid_dataset = load_dataset(mode="val", transform=preprocessing_transform)

model = UNet3D(in_channels=1 if args.channel else 2).cuda()
model.load_state_dict(torch.load(f"{args.model_folder}/checkpoints/epoch_{args.epoch}.pth"))
model.eval()
metric = Dice().cuda()

for i, subject in enumerate(valid_dataset):
    logger.info(f"Loading data {subject['name']}")
    
    sampler = tio.GridSampler(subject, patch_size=96, patch_overlap=12,)
    patch_loader = tio.SubjectsLoader(sampler, batch_size=1)
    aggregator = tio.inference.GridAggregator(sampler)

    with torch.no_grad():
        for patch_batch in patch_loader:
            if args.channel == "flair":
                image = patch_batch["flair"]["data"].cuda()
            elif args.channel == "dwi":
                image = patch_batch["dwi"]["data"].cuda()
            else:
                image = torch.cat((patch_batch["flair"]["data"], patch_batch["dwi"]["data"]), dim=1).cuda()

            prediction = model(image)
            aggregator.add_batch(prediction, patch_batch[tio.LOCATION])

    output_tensor = aggregator.get_output_tensor()
    if args.probabilities:
        tio.ScalarImage(tensor=output_tensor, affine=subject["flair"].affine).save(f"{args.model_folder}/predictions/{subject['name']}_probabilities.nii.gz")
    output_tensor = torch.sigmoid(output_tensor).round()

    dice = metric(output_tensor, subject["label"]["data"].round())
    logger.info(f"Dice: {dice:.4f}")

    logger.info(f"Saving subject {i+1}/{len(valid_dataset)}")
    labelmap = tio.LabelMap(tensor=output_tensor, affine=subject["flair"].affine)
    labelmap = subject.get_inverse_transform(warn=False)(labelmap)
    labelmap.save(f"{args.model_folder}/predictions/{subject['name']}.nii.gz")