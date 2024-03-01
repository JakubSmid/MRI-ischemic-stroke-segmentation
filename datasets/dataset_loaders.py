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

            self.flairs.append(f"{dataset_folder}/{patient_folder}/{flair}/{flair}.nii")
            self.dwis.append(f"{dataset_folder}/{patient_folder}/{dwi}/{dwi}.nii")
            self.masks.append(f"{dataset_folder}/{patient_folder}/{mask}/{mask}.nii")
            self.names.append(f"{patient_folder}")

class ISLES2022():
    def __init__(self, dataset_folder = "datasets/ISLES-2022/"):
        self.dataset_folder = dataset_folder
        self.flairs = []
        self.dwis = []
        self.masks = []
        self.names = []

        sub_strokecases = [f"sub-strokecase{i:04d}" for i in range(1,251)]
        self.dataset_folder = dataset_folder
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
        self.patient_folders = []
        self.anats = []

        for patient_folder in os.listdir(dataset_folder):
            # skip corrupted
            if patient_folder == "2282310":
                continue
            for anat in os.listdir(f'{dataset_folder}/{patient_folder}'):
                # skip corrupted data
                corrupted = ["Anat_20211008","Anat_20230109"]
                if anat in corrupted:
                    continue

                self.flairs.append(f"{dataset_folder}/{patient_folder}/{anat}/rFlair.nii")
                self.dwis.append(f"{dataset_folder}/{patient_folder}/{anat}/rDWI2.nii")
                self.masks.append(f"{dataset_folder}/{patient_folder}/{anat}/Leze_FLAIR_DWI2.nrrd")
                self.names.append(f"{patient_folder}/{anat}")
                self.BETmasks.append(f"{dataset_folder}/{patient_folder}/{anat}/BETmask.nii")
                self.patient_folders.append(patient_folder)
                self.anats.append(anat)