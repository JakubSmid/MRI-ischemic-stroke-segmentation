import numpy as np
import matplotlib.pyplot as plt

origin = np.array([[0],
                   [0]])

basis = np.array([
    [1, 0],
    [0, 1]
])

rectangle = np.array([
    [0, 0, 4, 4],
    [0, 4, 4, 0]
    ])

affine = np.array([
    [2, 0, 0],
    [0, 2, 0],
    [0, 0, 1]
    ])

def plot_rectangle(points, color, label):
    plt.plot([points[0, 0], points[0, 1], points[0, 2], points[0, 3], points[0, 0]],
             [points[1, 0], points[1, 1], points[1, 2], points[1, 3], points[1, 0]],
             color=color, label=label)
    plt.plot(points[0], points[1], 'o', color=color)

def plot_basis(basis, origin):
    plt.arrow(origin[0][0], origin[1][0], basis[0][0] - origin[0][0], basis[1][0] - origin[1][0],
          length_includes_head=True, head_width=0.1, color='red', zorder=2)
    plt.arrow(origin[0][0], origin[1][0], basis[0][1] - origin[0][0], basis[1][1] - origin[1][0],
          length_includes_head=True, head_width=0.1, color='green', zorder=2)

transformed = np.dot(affine, np.vstack((rectangle, np.ones(rectangle.shape[1]))))[:2]
transformed_base = np.dot(affine, np.vstack((basis, np.ones(basis.shape[1]))))[:2]
transformed_origin = np.dot(affine, np.vstack((origin, np.ones(origin.shape[1]))))[:2]

plot_rectangle(rectangle, 'blue', f'Vzor')
plot_rectangle(transformed, 'orange', f'Obraz')
plot_basis(basis, origin)
plot_basis(transformed_base, transformed_origin)

plt.title(f'Afinní zobrazení')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid()
plt.axis('square')

plt.show()