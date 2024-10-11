import os
import logging
import time

import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adamax
from torch.utils.tensorboard import SummaryWriter

from torchmetrics.classification import Dice

import torchio as tio

from model import UNet3D
from dataset import load_dataset

model_name = f"LabelSampler_augmentation2_{time.strftime('%Y%m%d_%H%M%S')}"

def log_images(subjects, writer, tag, epoch):
    batch_size = len(subjects["name"])
    for i in range(batch_size):
        idx = torch.argmax(label[i][0].sum(dim=(0,2)))
        rescale = tio.RescaleIntensity(out_min_max=(0, 1))
        image = torch.concat((rescale(subjects["flair"]["data"][i])[0][:, idx], rescale(subjects["dwi"]["data"][i])[0][:, idx],
                                label[i][0][:, idx].cpu(), segmentation[i][0][:, idx].cpu()), dim=1)
        writer.add_image(f"{tag}/{subjects['name'][i]}", image, global_step=epoch, dataformats="HW")

os.makedirs(f"output/logs/{model_name}", exist_ok=True)
os.makedirs(f"output/{model_name}/checkpoints", exist_ok=True)

# set up logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s [%(name)s] %(message)s",
    datefmt="%d/%b/%Y %H:%M:%S",
    handlers=[
        logging.FileHandler(f"output/{model_name}/{model_name}.log", mode='w'),
        logging.StreamHandler()
    ])

logger = logging.getLogger(__name__)
writer = SummaryWriter(log_dir=f"output/logs/{model_name}")

preprocessing_transform = tio.Compose([
    tio.Pad((48, 48, 48)),
    tio.ZNormalization(),
])
transform = tio.Compose([
    preprocessing_transform,
    tio.RandomAffine(scales=0.1, degrees=45, default_pad_value="otsu"),
    tio.RandomBlur((0, 0.4)),
    tio.RandomNoise(std=0.05),
])

# load dataloaders and model
train_dataset = load_dataset(mode="train", transform=transform, exclude_empty=True)
valid_dataset = load_dataset(mode="val", transform=preprocessing_transform, exclude_empty=True)

sampler = tio.LabelSampler(patch_size=96, label_name="label")

train_patches_queue = tio.Queue(
    train_dataset,
    max_length=300,
    samples_per_volume=10,
    sampler=sampler,
    num_workers=12
)

val_patches_queue = tio.Queue(
    valid_dataset,
    max_length=28,
    samples_per_volume=1,
    sampler=sampler,
    num_workers=12
)
logger.info(f"Max train RAM usage: {train_patches_queue.get_max_memory_pretty()} MB")

train_dataloader = tio.SubjectsLoader(train_patches_queue, batch_size=2)
valid_dataloader = tio.SubjectsLoader(val_patches_queue, batch_size=2)

# load optimizer and metrics
model = UNet3D(in_channels=2).cuda()
metric = Dice().cuda()
optimizer = Adamax(params=model.parameters())

for epoch in range(20):
    logger.info(f"---------------------------------------------- Starting new epoch {epoch+1} ----------------------------------------------")
    train_loss = 0
    train_dice = 0
    valid_loss = 0
    valid_dice = 0

    # training
    model.train()
    train_start_time = time.time()
    for i, subjects in enumerate(train_dataloader):
        logger.info(f"Loading new batch {i+1}/{len(train_dataloader)}")
        logger.info(f"Loading data {subjects['name']}")

        # load data
        images = torch.cat((subjects["flair"]['data'], subjects["dwi"]['data']), dim=1).cuda() # [B, C, W, H, D]
        label = subjects['label']['data'].round().cuda()

        # update loss positive weight
        loss_fn = BCEWithLogitsLoss(pos_weight=(label==0.).sum()/label.sum())

        optimizer.zero_grad()
        prediction = model(images)
        loss = loss_fn(prediction, label)
        loss.backward()
        optimizer.step()
        
        # log statistics
        segmentation = torch.sigmoid(prediction).round()
        train_loss += loss.item()
        dice = metric(segmentation, label.to(torch.int))
        train_dice += dice
        logger.info(f"Training Dice: {dice:.4f}")
        logger.info(f"Average time per subject: {(time.time() - train_start_time)/(i+1):.02f} s")

        # log images
        # log_images(subjects, writer, "Train", epoch)

    # calculate statistics
    train_loss = train_loss / len(train_dataloader)
    train_dice = train_dice / len(train_dataloader)
    
    # log training results
    logger.info(f"--------------- Training finished - printing results ---------------")
    logger.info(f"Training Loss: {train_loss} \t Training Dice: {train_dice:.4f}")
    logger.info(f"Training time: {time.time() - train_start_time:.2f} s \t Aveage time per subject: {(time.time() - train_start_time)/len(train_dataloader):.02f} s")
    writer.add_scalar("Loss/Train", train_loss, epoch)
    writer.add_scalar("Dice/Train", train_dice, epoch)

    logger.info(f"--------------- Starting validation ---------------")
    model.eval()
    valid_start_time = time.time()
    with torch.no_grad():
        for i, subjects in enumerate(valid_dataloader):
            logger.info(f"Loading new batch {i+1}/{len(valid_dataloader)}")
            logger.info(f"Loading data {subjects['name']}")
            
            # load data
            images = torch.cat((subjects["flair"]['data'], subjects["dwi"]['data']), dim=1).cuda()
            label = subjects['label']['data'].cuda()
            prediction = model(images)

            # update loss positive weight
            loss_fn = BCEWithLogitsLoss(pos_weight=(label==0.).sum()/label.sum())   
            loss = loss_fn(prediction, label.type(torch.FloatTensor).cuda())
            
            segmentation = torch.sigmoid(prediction).round()
            valid_loss += loss.item()
            dice = metric(segmentation, label)
            valid_dice += dice
            logger.info(f"Validation Dice: {dice:.4f}")

            # log images
            # log_images(subjects, writer, "Validation", epoch)
            
    # calculate statistics
    valid_loss = valid_loss / len(valid_dataloader)
    valid_dice = valid_dice / len(valid_dataloader)
    
    # log validation results
    logger.info(f"--------------- Validation finished - printing results ---------------")
    logger.info(f"Validation Loss: {valid_loss} \t Validation Dice: {valid_dice:.4f}")
    logger.info(f"Validation time: {time.time() - valid_start_time:.2f} s \t Aveage time per subject: {(time.time() - valid_start_time)/len(valid_dataloader):.02f} s")
    writer.add_scalar("Loss/Validation", valid_loss, epoch)
    writer.add_scalar("Dice/Validation", valid_dice, epoch)
    
    if (epoch+1)%5 == 0:
        logger.info(f"Saving model epoch_{str(epoch+1).format('03')}.pth")
        torch.save(model.state_dict(), f"output/{model_name}/checkpoints/epoch_{str(epoch+1).format('03')}.pth")
