import numpy as np
import nibabel as nib
import nibabel.processing
import ants
import time
import datasets.dataset_loaders as dataset_loaders

def timings(subj):
    # load data
    nib_load_time = time.time()
    flair = nib.load(subj.flair)
    dwi = nib.load(subj.dwi)
    mask = nib.load(subj.BETmask)
    nib_load_time = time.time() - nib_load_time

    # resample dwi to flair
    nib_resample_time = time.time()
    flair = nibabel.processing.conform(flair, out_shape=(200,200,200), voxel_size=(1,1,1))
    dwi = nibabel.processing.resample_from_to(dwi, flair)
    mask = nibabel.processing.resample_from_to(mask, flair)
    nib_resample_time = time.time() - nib_resample_time

    # apply mask
    nib_mask_time = time.time()
    flair = nib.nifti1.Nifti1Image(flair.get_fdata() * mask.get_fdata().astype(np.int32), flair.affine, flair.header)
    dwi = nib.nifti1.Nifti1Image(dwi.get_fdata() * mask.get_fdata().astype(np.int32), dwi.affine, dwi.header)
    nib_mask_time = time.time() - nib_mask_time

    # save and print results
    nib.save(flair, "results/flair_nib.nii.gz")
    nib.save(dwi, "results/dwi_nib.nii.gz")
    print(f"Nibabel load time: {nib_load_time:.3f} s")
    print(f"Nibabel resample time: {nib_resample_time:.3f} s")
    print(f"Nibabel mask time: {nib_mask_time:.3f} s")

    # load data
    ants_load_time = time.time()
    flair = ants.image_read(subj.flair)
    dwi = ants.image_read(subj.dwi)
    mask = ants.image_read(subj.BETmask)
    ants_load_time = time.time() - ants_load_time

    # resample dwi to flair
    ants_resample_time = time.time()
    flair = ants.crop_image(flair, mask)
    dwi = ants.crop_image(dwi, mask)

    flair = ants.resample_image(flair, (1.0, 1.0, 1.0), use_voxels=False)
    flair = ants.pad_image(flair, (200, 200, 200))

    dwi = ants.resample_image_to_target(dwi, flair)
    mask = ants.resample_image_to_target(mask, flair)
    ants_resample_time = time.time() - ants_resample_time

    # apply mask
    ants_mask_time = time.time()
    flair = ants.mask_image(flair, mask)
    dwi = ants.mask_image(dwi, mask)
    ants_mask_time = time.time() - ants_mask_time

    # save and print results
    ants.image_write(flair, "results/flair_ants.nii.gz")
    ants.image_write(dwi, "results/dwi_ants.nii.gz")
    print(f"ANTs load time: {ants_load_time:.3f} s")
    print(f"ANTs resample time: {ants_resample_time:.3f} s")
    print(f"ANTs mask time: {ants_mask_time:.3f} s")

def nib_load_time(subj, rep=10):
    start = time.time()
    for _ in range(rep):
        nib.load(subj.flair)
    print(f"Nibabel load time {rep} trials: {time.time() - start:.3f} s")

def ants_load_time(subj, rep=10):
    start = time.time()
    for _ in range(rep):
        ants.image_read(subj.flair)
    print(f"ANTs load time {rep} trials: {time.time() - start:.3f} s")

def nib_2_ants(subj, rep=10):
    start = time.time()
    for _ in range(rep):
        img = nib.load(subj.flair)
        ants.nifti_to_ants(img)
    print(f"Nifti to ANTs {rep} trials: {time.time() - start:.3f} s")

if __name__ == "__main__":
    subj = dataset_loaders.Motol()[0]
    timings(subj)
    nib_load_time(subj)
    ants_load_time(subj)
    nib_2_ants(subj)