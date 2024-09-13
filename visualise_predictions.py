import numpy as np
import ants
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches
import datasets.dataset_loaders as dataset_loaders
import datasets.utils as utils
import argparse
import os

def maximum_area(mask: ants.ants_image.ANTsImage) -> int:
    mask_np = mask.numpy()
    return np.argmax(mask_np.sum(axis=0).sum(axis=0))

def plot_image(image: ants.ants_image.ANTsImage, slice: int, ax: plt.Axes):
    view = image.numpy()[:,:,slice]
    view = np.flip(view, axis=0)
    view = np.rot90(view, -1)
    aspect = image.spacing[2]/image.spacing[0]

    upper_quartile = np.percentile(view, 75)
    lower_quartile = np.percentile(view, 25)
    iqr = upper_quartile - lower_quartile

    ax.set_axis_off()
    ax.imshow(view, cmap="gray", vmax=min(upper_quartile + iqr*1.5, view.max()),
              aspect=aspect,
              extent=[image.shape[0], 0, image.shape[2], 0]
              )

def plot_label(label: ants.ants_image.ANTsImage, slice: int, ax: plt.Axes):
    mask = label.numpy()[:,:,slice]
    mask = np.flip(mask, axis=0)
    mask = np.rot90(mask, -1)
    aspect = label.spacing[2]/label.spacing[0]

    mask = np.ma.masked_where(mask == 0, mask)
    ax.set_axis_off()
    ax.imshow(mask, cmap="Set1", vmax=10, interpolation="none",
              aspect=aspect,
              extent=[label.shape[0], 0, label.shape[2], 0],
              alpha=0.5)

def plot_sheet():
    # load dataset
    dataset = dataset_loaders.Motol()
    dataset = [subj for subj in dataset if subj.name in args.images]

    # calculate number of rows and columns
    nrows=np.ceil(np.sqrt(len(dataset))).astype(int)
    ncols=np.ceil(len(dataset)/nrows).astype(int)

    # create figure
    fig, axs = plt.subplots(nrows, ncols, figsize=(8.5, 10))
    plt.rcParams['axes.titley'] = 0.0 # adjust y position of title
    plt.rcParams['axes.titlesize'] = 8
    plt.rcParams['axes.titlecolor'] = "white"

    for i, subj in enumerate(dataset):
        print(f"Plotting {i+1}/{len(dataset)}: {subj.name}")
        if ncols == 1:
            if nrows == 1:
                ax = axs
            else:
                ax = axs[i]
        else:
            ax = axs[i // ncols, i % ncols]

        # load data and reorient to RAS
        subj.load_data()
        subj.flair = ants.reorient_image2(subj.flair)
        subj.label = ants.reorient_image2(subj.label)

        # load prediction
        pred = ants.image_read(f"{args.pred_folder}/{subj.name}.nii.gz")
        pred = utils.resample_label_to_target(pred, subj.label.astype("float32"))
        slice = maximum_area(subj.label)

        # merge expert and prediction
        new_label = np.zeros(subj.label.shape)
        new_label[subj.label.numpy()==1] = 1
        new_label[pred.numpy()==1] = 5
        new_label[(subj.label.numpy()==1) & (pred.numpy()==1)] = 3
        label = ants.new_image_like(subj.label, new_label)
        patches = [mpatches.Patch(color=matplotlib.colormaps["Set1"](0.1), label="Pouze expertní segmentace"),
                mpatches.Patch(color=matplotlib.colormaps["Set1"](0.5), label="Pouze predikce"),
                mpatches.Patch(color=matplotlib.colormaps["Set1"](0.3), label="Průnik segmentací")]

        # plot slice
        plot_image(subj.flair, slice, ax)
        plot_label(label, slice, ax)

        ax.set_title(f"{subj.name}\ny={subj.flair.shape[2]-slice}")
        subj.free_data()

    # remove unused axes
    for i in range(len(dataset), nrows*ncols):
        fig.delaxes(axs[i // ncols, i % ncols])

    if len(dataset) == nrows*ncols:
        fig.legend(handles=patches, loc="lower center", ncol=3)
        fig.subplots_adjust(top=1, bottom=0.05, left=0, right=1, wspace=0, hspace=0)
    else:
        fig.legend(handles=patches, loc="lower right")
        fig.subplots_adjust(top=1, bottom=0, left=0, right=1, wspace=0, hspace=0)
    fig.savefig(args.output, dpi=300, bbox_inches="tight", pad_inches=0.01)

def plot_four():
    # load dataset
    dataset = dataset_loaders.Motol()

    for i, subj in enumerate(dataset):
        print(f"Plotting {i+1}/{len(dataset)}: {subj.name}")
        fig, axs = plt.subplots(2, 2, figsize=(8.01, 10))

        # load data and reorient to RAS
        subj.load_data()
        subj.flair = ants.reorient_image2(subj.flair)
        subj.dwi = ants.reorient_image2(subj.dwi)
        subj.label = ants.reorient_image2(subj.label)

        # load prediction
        pred = ants.image_read(f"{args.pred_folder}/{subj.name}.nii.gz")
        pred = utils.resample_label_to_target(pred, subj.label.astype("float32"))
        slice = maximum_area(subj.label)

        # merge expert and prediction
        new_label = np.zeros(subj.label.shape)
        new_label[subj.label.numpy()==1] = 1
        new_label[pred.numpy()==1] = 5
        new_label[(subj.label.numpy()==1) & (pred.numpy()==1)] = 3
        label = ants.new_image_like(subj.label, new_label)
        patches = [mpatches.Patch(color=matplotlib.colormaps["Set1"](0.1), label="Pouze expertní segmentace"),
                mpatches.Patch(color=matplotlib.colormaps["Set1"](0.5), label="Pouze predikce"),
                mpatches.Patch(color=matplotlib.colormaps["Set1"](0.3), label="Průnik segmentací")]

        # plot slice
        axs[0,0].text(0.05, 0.95, "FLAIR", size=14, color="white", ha="left", va="top", transform=axs[0,0].transAxes)
        axs[0,1].text(0.05, 0.95, "FLAIR", size=14, color="white", ha="left", va="top", transform=axs[0,1].transAxes)
        axs[1,0].text(0.05, 0.95, "DWI", size=14, color="white", ha="left", va="top", transform=axs[1,0].transAxes)
        axs[1,1].text(0.05, 0.95, "DWI", size=14, color="white", ha="left", va="top", transform=axs[1,1].transAxes)

        plot_image(subj.flair, slice, axs[0,0])
        plot_image(subj.flair, slice, axs[0,1])
        plot_label(label, slice, axs[0,1])

        plot_image(subj.dwi, slice, axs[1,0])
        plot_image(subj.dwi, slice, axs[1,1])
        plot_label(label, slice, axs[1,1])

        fig.suptitle(f"{subj.name}\ny={subj.flair.shape[2]-slice}")
        fig.legend(handles=patches, loc="lower center", ncol=3)
        fig.subplots_adjust(top=0.94, bottom=0.05, left=0, right=1, wspace=0, hspace=0)

        os.makedirs(args.output, exist_ok=True)
        fig.savefig(f"{args.output}/{subj.name}.png", dpi=300, bbox_inches="tight", pad_inches=0.01)

        subj.free_data()

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("mode", choices=["sheet", "four"], help="mode")
    args.add_argument("pred_folder", help="folder with predictions")
    args.add_argument("output", help="output file or folder")
    args.add_argument("images", nargs='*', help="images to plot")
    args = args.parse_args()

    if args.mode == "sheet":
        plot_sheet()
    elif args.mode == "four":
        plot_four()