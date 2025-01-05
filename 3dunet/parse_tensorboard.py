from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import numpy as np
import os 

class ParseTensorboard:
    def __init__(self, folder: str):
        ea = event_accumulator.EventAccumulator(folder)
        ea.Reload()
        self.train_dice = np.array([i.value for i in ea.Scalars("Dice/Train")])
        self._train_dice_steps = np.array([i.step for i in ea.Scalars("Dice/Train")])+1
        
        self.val_dice = np.array([i.value for i in ea.Scalars("Dice/Validation")])
        self._val_dice_steps = np.array([i.step for i in ea.Scalars("Dice/Validation")])+1

    def plot(self, title="Výsledky", save=False):
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.plot(self._train_dice_steps, self.train_dice, label="Trénovací Dice koeficient")
        ax.plot(self._val_dice_steps, self.val_dice, label="Validační Dice koeficient")

        ax.grid()
        ax.legend(loc="upper left")
        ax.set_ylim(0, 1)
        ax.set_xticks(np.arange(0, max(self._train_dice_steps)+2, 2))
        ax.set_xlabel("Epocha")
        ax.set_ylabel("Dice koeficient")
        ax.title.set_text("Průměrné Dice koeficienty zaznamenané pro každou epochu")
    

        fig.suptitle(title, fontsize=18, fontweight="medium")
        fig.tight_layout()

        if save:
            fig.savefig("3dunet_" + title + ".png", dpi=300)

if __name__ == "__main__":
    # load folders
    data = "3dunet/output/logs/LabelSampler_20241022_234412/events.out.tfevents.1729633452.cmpgrid-79.3955324.0"
    ParseTensorboard(f"{data}").plot(title="LabelSampler", save=True)

    plt.show()