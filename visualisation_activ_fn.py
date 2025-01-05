import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def relu(x):
    return np.maximum(0, x)
def leaky_relu(x, alpha=0.05):
    return np.maximum(alpha * x, x)

fig, ax = plt.subplots(1, 3, figsize=(12, 4))

x = np.linspace(-10, 10, 100)
ax[0].plot(x, sigmoid(x), label="Sigmoida", color="b")
ax[0].set_xlabel("x")
ax[0].set_ylabel("y")
ax[0].title.set_text("Sigmoida")
ax[0].grid()

x = np.linspace(-1, 1, 100)
ax[1].plot(x, relu(x), label="ReLU", color="g")
ax[1].set_xlabel("x")
ax[1].set_ylabel("y")
ax[1].title.set_text("ReLU")
ax[1].grid()

ax[2].plot(x, leaky_relu(x), label="Leaky ReLU", color="r")
ax[2].set_xlabel("x")
ax[2].set_ylabel("y")
ax[2].title.set_text("Leaky ReLU")
ax[2].grid()

plt.show()