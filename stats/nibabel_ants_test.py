import numpy as np
import nibabel as nib
import nibabel.processing
import ants
import time
import datasets.dataset_loaders as dataset_loaders

def time_nifti_to_numpy(N_TRIALS):
    """
    Function from ANTsPy/tests/timings_io.py

    On a Macbook Pro
    ---------------
    1 Trial:
    NIBABEL TIME: 2.902 seconds
    ITK TIME: 4.375 seconds
    ANTS TIME: 1.713 seconds

    20 Trials:
    NIBABEL TIME: 56.121 seconds
    ITK TIME: 28.780 seconds
    ANTS TIME: 33.601 seconds


    Times how fast a framework can read a nifti file and convert it to numpy
    """
    img_paths = [ants.get_ants_data('mni')]*10
    
    def test_nibabel():
        for img_path in img_paths:
            array = np.asanyarray(nib.load(img_path).dataobj)

    def test_ants():
        for img_path in img_paths:
            array = ants.image_read(img_path, pixeltype='float').numpy()

    nib_start = time.time()
    for i in range(N_TRIALS):
        test_nibabel()
    nib_end = time.time()
    print('NIBABEL TIME: %.3f seconds' % (nib_end-nib_start))

    ants_start = time.time()
    for i in range(N_TRIALS):
        test_ants()
    ants_end = time.time()
    print('ANTS TIME: %.3f seconds' % (ants_end-ants_start))

def timings(subj: dataset_loaders.Subject):
    """
    Times NiBabel and ANTs for loading, resampling and masking
    a subject from the Motol dataset.

    Parameters:
        subj (dataset_loaders.Subject): A subject from the Motol dataset.
    """
    # load data
    nib_load_time = time.time()
    flair = nib.load(subj.flair)
    dwi = nib.load(subj.dwi)
    mask = nib.load(subj.BETmask)
    nib_load_time = time.time() - nib_load_time

    # resample dwi to flair
    nib_resample_time = time.time()
    flair = nibabel.processing.conform(flair, out_shape=(200,200,200), voxel_size=(1,1,1), order=1)
    dwi = nibabel.processing.resample_from_to(dwi, flair, order=1)
    mask = nibabel.processing.resample_from_to(mask, flair, order=1)
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
    """
    Measure the time it takes to load an image using nibabel.

    Parameters:
        subj (dataset_loaders.Subject): A subject from the Motol dataset.
        rep (int, optional): Number of repetitions. Defaults to 10.
    """
    start = time.time()
    for _ in range(rep):
        nib.load(subj.flair).get_fdata()
    print(f"Nibabel load time {rep} trials: {time.time() - start:.3f} s")

def ants_load_time(subj, rep=10):
    """
    Measure the time it takes to load an image using ANTs.

    Parameters:
        subj (dataset_loaders.Subject): A subject from the Motol dataset.
        rep (int, optional): Number of repetitions. Defaults to 10.
    """
    start = time.time()
    for _ in range(rep):
        ants.image_read(subj.flair).numpy()
    print(f"ANTs load time {rep} trials: {time.time() - start:.3f} s")

def nib_2_ants(subj, rep=10):
    """
    Measure the time it takes to load and convert a NiBabel image to an ANTs image.

    Parameters:
        subj (dataset_loaders.Subject): A subject from the Motol dataset.
        rep (int, optional): Number of repetitions. Defaults to 10.
    """
    start = time.time()
    for _ in range(rep):
        img = nib.load(subj.flair)
        ants.nifti_to_ants(img).numpy()
    print(f"Nifti to ANTs {rep} trials: {time.time() - start:.3f} s")

if __name__ == "__main__":
    subj = dataset_loaders.Motol()[0]
    timings(subj)
    nib_load_time(subj)
    ants_load_time(subj)
    nib_2_ants(subj)
    
    print(f"\nScript from ANTs")
    time_nifti_to_numpy(20)