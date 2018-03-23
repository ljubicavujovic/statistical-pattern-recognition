import os
import numpy as np
import matplotlib.pyplot as plt
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
    plt.figure(figsize=(7, 7))
    plt.scatter(x[:, 0], x[:, 1], marker='o', c=y[:,0], s=25, edgecolor='k')
    plt.title("Samples")
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()


def hypotesis(x, theta, k):
    s = 0
    for j in range(theta.shape[1]):
        s += np.exp(np.dot(x, theta[:, j]))
    p = np.exp(np.dot(x, theta[:, k]))/s
    return p


def cost_function(x, y, theta):
    J = 0
    for i in range(len(x)):
        for k in range(theta.shape[1]):
            J += int(y[i] == k)*np.log(hypotesis(x[i, :], theta, k))
    return J


def gradient(x, y, theta):
    grad = np.zeros(theta.shape)
    for k in range(theta.shape[1]):
        for i in range(len(x)):
            grad[:, k] += x[i, :]*((y[i]==k) - hypotesis(x[i, :], theta, k))
    return grad


def soft_max(x, y, k=3, alpha=0.001, iterations=1000):
    z = np.ones((len(x), 1))
    X = np.concatenate((z, x), axis=1)
    theta = np.ones((X.shape[1], k))
    cost = np.zeros((iterations, 1))
    for i in range(iterations):
        theta += alpha*gradient(X, y, theta)
        cost[i] = cost_function(X, y, theta)
    return theta, cost


def predict(x, theta):
    X = np.concatenate((np.ones((len(x), 1)), x), axis=1)
    P = np.zeros((len(x), theta.shape[1]))
    for i in range(theta.shape[1]):
        P[:, i] = hypotesis(X, theta, i)
    prediction = np.zeros((len(x), 1))
    for i in range(len(x)):
        k = np.argmax(P[i, :])
        prediction[i] = k

    return prediction


def plot_cost_function(cost):
    plt.figure(figsize=(8, 8))
    plt.plot(cost)
    plt.title("Negative cost as fucntion  \n of number of iterations")
    plt.xlabel("Number of iterations")
    plt.ylabel("Negative cost J")
    plt.show()


def plot_line(x, y, theta):
    Z = predict(x, theta)
    missed = []
    label = []
    for i in range(len(x)):
        if Z[i] != y[i]:
            missed.append(x[i,: ])
            label.append(int(Z[i]))
    missed = np.array(missed)

    plt.figure(figsize=(8, 8))
    plt.scatter(x[np.where(y == 0)[0], 0], x[np.where(y == 0)[0], 1], marker='o', c='r', s=25, label="First class")
    plt.scatter(x[np.where(y == 1)[0], 0], x[np.where(y == 1)[0], 1], marker='o', c='y', s=25, label="Second class")
    plt.scatter(x[np.where(y == 2)[0], 0], x[np.where(y == 2)[0], 1], marker='o', c='b', s=25, label="Third class")

    plt.scatter(missed[:, 0], missed[:, 1], marker='*', linewidth=3.0, c='g', s=25, label='Misclassified samples')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()
    return 1 - len(missed)/len(y)


if __name__ == "__main__":
    x, y = read_data("softmaxData.txt")
    theta, c = soft_max(x, y)
    visualize_data(x, y)
    plot_cost_function(c)
    acc = plot_line(x, y, theta)
    print("**********Parameters*********")
    print(theta)
    print("**********Accuracy*********")
    print(acc)
