import os 

class ISLES2015():
    def __init__(self, dataset_folder = "datasets/SISS2015_Training/"):
        self.dataset_folder = dataset_folder
        self.flairs = []
        self.dwis = []
        self.masks = []
        self.names = []

        for patient_folder in range(1,29):
            scan_folders = os.listdir(f"datasets/SISS2015_Training/{patient_folder}")
            flair = next(filter(lambda x: "Flair" in x, scan_folders))
            dwi = next(filter(lambda x: "DWI" in x, scan_folders))
            mask = next(filter(lambda x: "OT" in x, scan_folders))

            self.flairs.append(f"{dataset_folder}/{patient_folder}/{flair}/{flair}.nii.gz")
            self.dwis.append(f"{dataset_folder}/{patient_folder}/{dwi}/{dwi}.nii.gz")
            self.masks.append(f"{dataset_folder}/{patient_folder}/{mask}/{mask}.nii.gz")
            self.names.append(f"{patient_folder}")

class ISLES2022():
    def __init__(self, dataset_folder = "datasets/ISLES-2022/"):
        self.dataset_folder = dataset_folder
        self.flairs = []
        self.dwis = []
        self.masks = []
        self.names = []

        sub_strokecases = [f"sub-strokecase{i:04d}" for i in range(1,251)]
        for sub_strokecase in sub_strokecases:
            self.dwis.append(f"{dataset_folder}/{sub_strokecase}/ses-0001/dwi/{sub_strokecase}_ses-0001_dwi.nii.gz")
            self.flairs.append(f"{dataset_folder}/{sub_strokecase}/ses-0001/anat/{sub_strokecase}_ses-0001_FLAIR.nii.gz")
            self.masks.append(f"{dataset_folder}/derivatives/{sub_strokecase}/ses-0001/{sub_strokecase}_ses-0001_msk.nii.gz")
            self.names.append(f"{sub_strokecase}")

class Motol():
    def __init__(self, dataset_folder = "datasets/Motol/"):
        self.dataset_folder = dataset_folder
        self.flairs = []
        self.dwis = []
        self.masks = []
        self.names = []
        self.BETmasks = []

        for patient_folder in os.listdir(dataset_folder):
            if patient_folder == "2290867":
                # skip corrupted patient
                continue
            for anat in os.listdir(f'{dataset_folder}/{patient_folder}'):
                self.flairs.append(f"{dataset_folder}/{patient_folder}/{anat}/rFlair.nii.gz")
                self.dwis.append(f"{dataset_folder}/{patient_folder}/{anat}/rDWI2.nii.gz")
                self.masks.append(f"{dataset_folder}/{patient_folder}/{anat}/Leze_FLAIR_DWI2.nrrd")
                self.names.append(f"{patient_folder}_{anat}")
                self.BETmasks.append(f"{dataset_folder}/{patient_folder}/{anat}/BETmask.nii.gz")