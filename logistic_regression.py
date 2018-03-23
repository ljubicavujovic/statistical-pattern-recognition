import os
import numpy as np
import matplotlib.pyplot as plt
from explore_data import read_data
from sklearn.metrics import confusion_matrix
plt.style.use('ggplot')


# def read_data(name, filepath=os.path.realpath(os.path.dirname(__file__))):
#     x = []
#     y = []
#     dataset_path = os.path.join(filepath, "data/" + name)
#     with open(dataset_path, 'r') as f:
#         for line in f:
#             numbers = [float(i) for i in line.split()]
#             x.append([numbers[0], numbers[1]])
#             y.append(numbers[2])

    # return np.array(x), np.array(y).reshape(len(y), 1)


def sigmoid(x, theta):
    return 1/(1 + np.exp(-np.dot(x, theta)))


def gradient(x, y, theta):
    dtheta = np.sum(np.multiply(y - sigmoid(x, theta), x), axis=0)
    return dtheta.reshape(theta.shape)


def cost_function(x, y, theta):
    h = sigmoid(x, theta)
    l = np.sum(np.multiply(y, np.log(h)) + np.multiply((1-y), np.log(1-h)))
    return l


def gradient_descent(x, y, alpha=0.001, iterations=1000):
    cost = np.zeros((iterations, 1))
    theta = np.ones((x.shape[1], 1))
    for i in range(iterations):
        theta += alpha*gradient(x, y, theta)
        cost[i] = cost_function(x, y, theta)
    return theta, cost


def gradient_n(x, y, theta, epsilon=1e-7):
    theta_plus = theta + np.array([[epsilon], [0], [0]])
    theta_minus = theta + np.array([[-epsilon], [0], [0]])
    J_plus = cost_function(x, y, theta_plus)
    J_minus = cost_function(x, y, theta_minus)
    dtheta_1 = (J_plus - J_minus)/(2*epsilon)

    theta_plus = theta + np.array([[0], [epsilon], [0]])
    theta_minus = theta + np.array([[0], [-epsilon], [0]])
    J_plus = cost_function(x, y, theta_plus)
    J_minus = cost_function(x, y, theta_minus)
    dtheta_2 = (J_plus - J_minus) / (2 * epsilon)

    theta_plus = theta + np.array([[0], [0], [epsilon]])
    theta_minus = theta + np.array([[0], [0], [-epsilon]])
    J_plus = cost_function(x, y, theta_plus)
    J_minus = cost_function(x, y, theta_minus)
    dtheta_3 = (J_plus - J_minus) / (2 * epsilon)

    return np.array([[dtheta_1], [dtheta_2], [dtheta_3]])


def predict(x, theta, treshold=0.5):
    x = np.concatenate((np.ones((len(x), 1)), x), axis=1)
    P = sigmoid(x, theta) > treshold
    return P


def calculate_metrics(x, y, theta):
    p = predict(x, theta)
    accuracy = np.sum(p == y)/len(y)
    cm = confusion_matrix(y, p)
    return {"accuracy": accuracy, "false positive": cm[0, 1], "false negative": cm[1, 0]}


def make_meshgrid(x, y, h=.01):
    x_min, x_max = x.min() - 0.5, x.max() + 0.5
    y_min, y_max = y.min() - 0.5, y.max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def visualize_data(x, y):
    plt.figure(figsize=(8, 8))
    plt.scatter(x[:, 0], x[:, 1], marker='o', c=y, s=25, edgecolor='k')
    plt.show()


def plot_cost_function(cost):

    plt.figure(figsize=(8, 8))
    plt.plot(cost[5:])
    plt.title("Negative cost as function \n of number of iterations for logistic regression")
    plt.xlabel("Number of iterations")
    plt.ylabel("value of negative cost J")
    plt.show()


def plot_line(x, y, theta):
    x0, x1 = make_meshgrid(x[:, 1], x[:, 2])
    Z = predict(np.c_[x0.ravel(), x1.ravel()], theta)
    Z = Z.reshape(x0.shape)

    plt.figure(figsize=(8, 8))
    plt.contour(x0, x1, Z, cmap=plt.cm.Paired)
    plt.scatter(x[np.where(y == 0)[0], 1], x[np.where(y == 0)[0], 2], marker='o', c='r', s=25, label="First class")
    plt.scatter(x[np.where(y == 1)[0], 1], x[np.where(y == 1)[0], 2], marker='o', c='b', s=25, label="Secont class")
    plt.title("Classification line and data")
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()
    return 0


def plot_roc_curve(x, y, iterations=10000):
    tresholds = np.linspace(0.1, 0.9, iterations)
    true_positive = np.zeros((iterations, 1))
    false_positive = np.zeros((iterations, 1))

    for i, treshold in enumerate(tresholds):
        p = predict(x, theta, treshold)
        cm = confusion_matrix(y, p)
        true_positive[i] = cm[1, 1]/np.sum(y == 1)
        false_positive[i] = cm[0, 1]/np.sum(y == 0)

    plt.figure(figsize=(8, 8))
    plt.plot(false_positive, true_positive)
    plt.title("ROC curve")
    plt.xlabel("False Positive")
    plt.ylabel("True Positive")
    plt.show()


if __name__ == "__main__":
    x, y = read_data('logregData.txt')
    z = np.ones((len(x), 1))
    X = np.concatenate((z, x), axis=1)

    theta, cost = gradient_descent(X, y, iterations=10000)
    plot_line(X, y, theta)
    print(theta)
    print(calculate_metrics(x, y, theta))
    plot_roc_curve(x, y)
    plot_cost_function(cost)
