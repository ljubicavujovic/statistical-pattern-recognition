import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
plt.style.use('ggplot')


def read_data(name, filepath=os.path.realpath(os.path.dirname(__file__))):
    x = []
    y = []
    dataset_path = os.path.join(filepath, "data/" + name)
    with open(dataset_path, 'r') as f:
        for line in f:
            numbers = [float(i) for i in line.split()]
            x.append([numbers[0], numbers[1]])
            y.append(numbers[2])
    return np.array(x), np.array(y).reshape(len(y), 1) - 1


def visualize_data(x, y):
    plt.figure(figsize=(8, 8))
    plt.scatter(x[np.where(y == 0)[0], 0], x[np.where(y == 0)[0], 1], marker='o', c='r', s=25, label="First Class")
    plt.scatter(x[np.where(y == 1)[0], 0], x[np.where(y == 1)[0], 1], marker='o', c='b', s=25, label="Second Class")
    plt.scatter(x[np.where(y == 2)[0], 0], x[np.where(y == 2)[0], 1], marker='o', c='y', s=25, label="Third Class")
    plt.title("Samples")
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()
    return 0


def naive_bayes(x, y, k=3):
    mean = np.zeros((x.shape[1], k))
    var = np.zeros((x.shape[1], k))
    for i in range(k):
        t = x[np.where(y == i), :][0]
        mean[0:, i] = t[:, 0].mean()
        mean[1:, i] = t[:, 1].mean()
        var[0, i] = t[:, 0].var()
        var[1, i] = t[:, 1].var()

    return mean, var


def gda(x, y, k=3):
    mean = np.zeros((x.shape[1], k))
    for i in range(k):
        t = x[np.where(y == i), :][0]
        mean[0:, i] = t[:, 0].mean()
        mean[1:, i] = t[:, 1].mean()
    cov = np.zeros((2, 2))
    for i in range(len(x)):
        tmp = x[i, :] - mean[:, int(y[i])]
        tmp = tmp.reshape(1, 2)
        cov += np.dot(tmp.T, tmp)
    cov /= len(x)
    return mean, cov


def gda_diff(x, y, k=3):
    mean = np.zeros((x.shape[1], k))
    cov = np.zeros((3, x.shape[1], x.shape[1]))
    covn = np.zeros((3, x.shape[1], x.shape[1]))

    for i in range(k):
        t = x[np.where(y == i), :][0]
        ty = y[np.where(y == i), :][0]
        mean[0:, i] = t[:, 0].mean()
        mean[1:, i] = t[:, 1].mean()
        cov[i] = np.cov(t.T, bias=True)
        for j in range(len(t)):
            tmp = t[j, :] - mean[:, int(ty[j])]
            tmp = tmp.reshape(1, 2)
            covn[i, :] += np.dot(tmp.T, tmp)
        covn[i, :] /= len(t)
    return mean, cov, covn


def predict_naive_bayes(x, mean, var):
    probability = np.zeros((len(x), 3))
    for i in range(3):
        m = mean[:, i]
        cov = np.diag(var[:, i])
        probability[:, i] = multivariate_normal.pdf(x, m, cov)
    prediction = np.argmax(probability, axis=1)
    return prediction


def predict_gda(x, mean, cov):
    probability = np.zeros((len(x), 3))
    for i in range(3):
        m = mean[:, i]
        probability[:, i] = multivariate_normal.pdf(x, m, cov)
    prediction = np.argmax(probability, axis=1)
    return prediction


def predict_gda_diff(x, mean, cov):
    probability = np.zeros((len(x), 3))
    for i in range(3):
        m = mean[:, i]
        covi = cov[i, :]
        probability[:, i] = multivariate_normal.pdf(x, m, covi)
    prediction = np.argmax(probability, axis=1)
    return prediction


def accuracy(prediction, y):
    prediction = prediction.reshape(len(y), 1)
    return np.sum(prediction == y)/len(y)


if __name__ == "__main__":
    x, y = read_data("data_genalg.txt")
    visualize_data(x, y)
    mean, var = naive_bayes(x, y)
    print("*** Naive Bayes ***")
    print(mean)
    print(var)
    p_nb = predict_naive_bayes(x, mean, var)
    print('*** Accuracy ***')
    print(accuracy(p_nb, y))

    print()

    mean, cov = gda(x, y)
    print("*** Gaussian Discriminant Analysis")
    print(mean)
    print(cov)
    p_gda = predict_gda(x, mean, cov)
    print('*** Accuracy ***')
    print(accuracy(p_gda, y))

    print()

    mean, cov, covn = gda_diff(x, y)
    print("*** Gaussian Discriminant Analysis with different covariances ***")
    print(mean)
    print(cov)
    print(covn)
    p_gdadiff = predict_gda_diff(x, mean, covn)
    print('*** Accuracy ***')
    print(accuracy(p_gdadiff, y))

