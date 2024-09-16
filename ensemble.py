import nibabel as nib
import numpy as np
import argparse
import os

def ensemble_func(data: np.ndarray) -> np.ndarray:
    """
    This function takes a 3D numpy array with shape (z, y, x) as input, 
    where z is the number of models, and (y, x) is the spatial dimension.
    It calculates the mean probability across all models, and then 
    applies a threshold at 0.5 to convert the probabilities to binary 
    segmentation masks. The resulting binary masks are then returned.

    Parameters:
        data (np.ndarray): A 4D numpy array with shape (z, y, x, c), where z is the number of models.

    Returns:
        np.ndarray: A 3D numpy array with shape (y, x) containing binary segmentation masks.
    """
    data = np.mean(data, axis=0)
    data = (data >= 0.5).astype(np.uint8)
    return data

def ensemble_nnUNet(input_folders: list[str], output_folder: str):
    """
    This function takes a list of folders as input, where each folder contains
    the predictions of a different model. The function then loads the data
    with probabilities from all the folders, ensembles the predictions, and
    saves the result to a new folder.

    Parameters:
        input_folders (list[str]): A list of paths to the input folders.
        output_folder (str): The path to the output folder.
    """
    for filename in os.listdir(args.input_folders[0]):
        # only consider .npz files
        if not filename.endswith(".npz"):
            continue

        # load input data from all input folders
        data = [np.load(os.path.join(folder, filename))["probabilities"][1] for folder in args.input_folders]
        metadata = np.load(os.path.join(args.input_folders[0], str(filename).replace(".npz", ".pkl")), allow_pickle=True)

        # prepare affine
        affine = np.zeros((4,4))
        affine[3, 3] = 1
        affine[:3, :3] = np.array(metadata["sitk_stuff"]["direction"]).reshape(3,3)
        affine[:3, 3] = metadata["sitk_stuff"]["origin"]
        affine[0,:]=-affine[0,:]
        affine[1,:]=-affine[1,:]
        
        data = ensemble_func(data)
        data = np.swapaxes(data, 0, 2)

        new_nifti = nib.Nifti1Image(data, affine)
        nib.save(new_nifti, os.path.join(args.output_folder, str(filename).replace(".npz", ".nii.gz")))
        print(f"Saved {filename} to {args.output_folder}")

def ensemble_deepmedic(input_folders, output_folder):
    """
    This function takes a list of folders as input, where each folder contains
    the predictions of a different model. The function then loads the data
    with probabilities from all the folders, ensembles the predictions, and
    saves the result to a new folder.

    Parameters:
        input_folders (list[str]): A list of paths to the input folders.
        output_folder (str): The path to the output folder.
    """
    for filename in os.listdir(args.input_folders[0]):
        # only consider ProbMapClass1.nii.gz files
        if not filename.endswith("_ProbMapClass1.nii.gz"):
            continue

        # load input data from all input folders
        data = [nib.load(os.path.join(folder, filename)).get_fdata() for folder in args.input_folders]
        
        data = ensemble_func(data)

        new_nifti = nib.Nifti1Image(data, nib.load(os.path.join(args.input_folders[0], filename)).affine)
        nib.save(new_nifti, os.path.join(args.output_folder, str(filename).replace("_ProbMapClass1.nii.gz", ".nii.gz")))
        print(f"Saved {filename} to {args.output_folder}")

if __name__ == "__main__":
    
    args = argparse.ArgumentParser()
    args.add_argument("mode", type=str, choices=["nnUNet", "deepmedic"])
    args.add_argument("output_folder", type=str)
    args.add_argument("input_folders", type=str, nargs="+")
    args = args.parse_args()

    if args.mode == "nnUNet":
        ensemble_nnUNet(args.input_folders, args.output_folder)
    elif args.mode == "deepmedic":
        ensemble_deepmedic(args.input_folders, args.output_folder)