from parse_tensorboard import ParseTensorboard
import matplotlib.pyplot as plt
import numpy as np
import os

to_combine = [f"deepmedic_workspace/output/tensorboard/{folder}" for folder in os.listdir("deepmedic_workspace/output/tensorboard")]

title = "Souhrn pro trénování a validaci"
average = 20
max_step = 300

fig, ax = plt.subplots(figsize=(8, 8))
ax.grid()
ax.set_ylim(0, 1)
ax.set_xlabel("Subepocha")
ax.set_ylabel("Dice koeficient")
ax.title.set_text("Klouzavý průměr (pro 20 subepoch) Dice koeficientu")

tbs = [ParseTensorboard(model) for model in to_combine]
for tb in tbs:
    name = to_combine[tbs.index(tb)].split("/")[-1]
    indicies = np.where(tb._train_dice_steps <= max_step)
    ax.plot(np.convolve(tb.train_dice[indicies], np.ones((average,))/average)[:-average+1], label=f"Trénovací DC {name}")

for tb in tbs:
    name = to_combine[tbs.index(tb)].split("/")[-1]
    indicies = np.where(tb._train_dice_steps <= max_step)
    ax.plot(np.convolve(tb.val_dice[indicies], np.ones((average,))/average)[:-average+1], label=f"Validační DC {name}")

ax.legend(loc="upper left")
fig.suptitle(title, fontsize=18, fontweight="medium")
fig.tight_layout()

fig.savefig(title + ".png", dpi=300)