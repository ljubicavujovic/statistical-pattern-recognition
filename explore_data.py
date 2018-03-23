import os
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')


def read_data(name, filepath=os.path.realpath(os.path.dirname(__file__))):
    x = []
    y = []
    dataset_path = os.path.join(filepath, 'data/' + name)
    with open(dataset_path) as f:
        for line in f:
            numbers = [float(i) for i in line.split()]
            x.append([numbers[0], numbers[1]])
            y.append(numbers[2])
    return np.array(x).reshape(len(x), 2), np.array(y).reshape(len(y), 1)


def visualize_data(x, y):
    plt.figure(figsize=(8, 8))
    plt.scatter(x[np.where(y == -1)[0], 0], x[np.where(y == -1)[0], 1], marker='o', c='r', s=25, label="First class")
    plt.scatter(x[np.where(y == 1)[0], 0], x[np.where(y == 1)[0], 1], marker='o', c='b', s=25, label="Second class")
    plt.title("Samples")
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()
    return 0


def make_meshgrid(x, y, h=.01):
    x_min, x_max = x.min() - 0.5, x.max() + 0.5
    y_min, y_max = y.min() - 0.5, y.max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_line(x, y, parameters, prediction):
    svx = parameters["support_vectors"][0]
    svy = parameters["support_vectors"][1]
    x0, x1 = make_meshgrid(x[:, 0], x[:, 1])
    prediction = prediction.reshape(x0.shape)

    plt.figure(figsize=(8, 8))
    plt.contour(x0, x1, prediction)
    plt.scatter(svx[np.where(svy == -1)[0], 0], svx[np.where(svy == -1)[0], 1], marker='*', c='m', s=100,
                label="Support vectors for first class", edgecolors='k')
    plt.scatter(svx[np.where(svy == 1)[0], 0], svx[np.where(svy == 1)[0], 1], marker='*', c='c', s=100,
                label="Support vectors for second class", edgecolors='k')
    plt.scatter(x[np.where(y == -1)[0], 0], x[np.where(y == -1)[0], 1], marker='o', c='m', s=20, label="First class")
    plt.scatter(x[np.where(y == 1)[0], 0], x[np.where(y == 1)[0], 1], marker='o', c='c', s=20, label="Second class")

    plt.title("Samples")
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()
    return 0