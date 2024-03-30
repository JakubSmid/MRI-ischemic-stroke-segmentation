from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import numpy as np
import os 

class ParseTensorboard:
    def __init__(self, folder: str):
        train = f"{folder}/train"
        val = f"{folder}/val"

        ea = event_accumulator.EventAccumulator(train)
        ea.Reload()
        self.train_dice = np.array([i.value for i in ea.Scalars("samples: Dice/Class-0")])
        self._train_dice_steps = np.array([i.step for i in ea.Scalars("samples: Dice/Class-0")])
        
        ea = event_accumulator.EventAccumulator(val)
        ea.Reload()
        self.val_dice = np.array([i.value for i in ea.Scalars("samples: Dice/Class-0")])
        self._val_dice_steps = np.array([i.step for i in ea.Scalars("samples: Dice/Class-0")])

        self.whole_scan = np.array([i.value for i in ea.Scalars("whole scans: Dice3 (Prediction VS Truth, both within ROI mask)/Class-0")])
        self._whole_scan_steps = np.array([i.step for i in ea.Scalars("whole scans: Dice3 (Prediction VS Truth, both within ROI mask)/Class-0")])

    def plot(self, title="Výsledky", save=False, max_step=300, average=10):
        fig, ax = plt.subplots(2, 1, figsize=(10, 8))
        indicies = np.where(self._train_dice_steps <= max_step)
        if average:
            ax[0].plot(self.train_dice[indicies], alpha=.3)
            ax[0].plot(np.convolve(self.train_dice[indicies], np.ones((average,))/average)[:-average+1], label=f"Klouzavý průměr Dice koeficientu na trénovacím datasetu", color="b")
        else:
            ax[0].plot(self.train_dice[indicies], label="Trénovací Dice koeficient")

        indicies = np.where(self._train_dice_steps <= max_step)
        if average:
            ax[0].plot(self.val_dice[indicies], alpha=.3)
            ax[0].plot(np.convolve(self.val_dice[indicies], np.ones((average,))/average)[:-average+1], label=f"Klouzavý průměr Dice koeficientu na validačním datasetu", color="orange")
        else:
            ax[0].plot(self.val_dice[indicies], label="Validační Dice koeficient")

        ax[0].grid()
        ax[0].legend(loc="upper left")
        ax[0].set_ylim(0, 1)
        ax[0].set_xlabel("Subepocha")
        ax[0].set_ylabel("Dice koeficient")
        ax[0].title.set_text("Průměrné Dice koeficienty zaznamenané pro každou subepochu")
        
        indicies = np.where(self._whole_scan_steps <= max_step)
        ax[1].plot(self._whole_scan_steps[indicies], self.whole_scan[indicies], label="Validační Dice koeficient na celých skenech", marker=".")
        ax[1].grid()
        ax[1].set_ylim(0, 1)
        ax[1].set_xlabel("Subepocha")
        ax[1].set_ylabel("Dice koeficient")
        ax[1].title.set_text("Průměrný Dice koeficient počítaný na celých skenech")

        fig.suptitle(title, fontsize=18, fontweight="medium")
        fig.tight_layout()

        if save:
            fig.savefig(title + ".png", dpi=300)

if __name__ == "__main__":
    # load folders
    data = os.listdir("deepmedic_workspace/output/tensorboard")
    for folder in data:
        ParseTensorboard(f"deepmedic_workspace/output/tensorboard/{folder}").plot(title=folder, save=True)

    #plt.show()