import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from explore_data import read_data
from matplotlib.ticker import LinearLocator, FormatStrFormatter
plt.style.use('ggplot')


def read_data(filepath=os.path.realpath(os.path.dirname(__file__))):
    x = []
    y = []
    dataset_path = os.path.join(filepath, 'data/data.txt')
    with open(dataset_path, 'r') as f:
        for line in f:
            x.append(float(line.split()[0]))
            y.append(float(line.split()[1]))
    x = np.array(x).reshape(len(x), 1)
    y = np.array(y).reshape(len(y), 1)
    return x, y


def pseudo_inversion(x, y):
    z = np.ones((len(x), 1))
    X = np.concatenate((z, x), axis=1)
    theta = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))

    min_x = np.min(x)
    max_x = np.max(x)
    points = 1000
    ax = np.linspace(min_x, max_x, points).reshape(points, 1)
    z = np.ones((points, 1))
    line = np.dot(np.concatenate((z, ax), axis=1), theta)
    plt.figure(figsize=(8, 8))
    plt.plot(x, y, "ro")
    plt.plot(ax, line, "b-")
    plt.title("Regression line with pseudoinversion method")
    plt.xlabel("x")
    plt.ylabel("y = theta1 +  x*theta2")
    plt.savefig("images\pseudo.png")
    plt.show()
    return theta


def cost_function(theta, x, y):
    z = np.ones((len(x), 1))
    x = x.reshape((len(x), 1))
    J = 0.5*np.sum((np.dot(np.concatenate((z, x), axis=1), theta) - y)**2)
    return J


def gradient(theta, x, y, epsilon=1e-7):
    theta_plus = theta + np.array([[epsilon], [0]])
    theta_minus = theta + np.array([[-epsilon], [0]])
    J_plus = cost_function(theta_plus, x, y)
    J_minus = cost_function(theta_minus, x, y)
    dtheta_1 = (J_plus - J_minus)/(2*epsilon)

    theta_plus = theta + np.array([[0], [epsilon]])
    theta_minus = theta + np.array([[0], [-epsilon]])
    J_plus = cost_function(theta_plus, x, y)
    J_minus = cost_function(theta_minus, x, y)
    dtheta_2 = (J_plus - J_minus) / (2 * epsilon)

    return np.array([[dtheta_1], [dtheta_2]])


def gradient_descent(start_theta, x, y, alfa, iterations):
    theta = start_theta
    thetas = np.zeros((iterations, 2))
    cost = np.zeros((iterations, 1))
    for i in range(iterations):
        cost[i] = cost_function(theta, x, y)
        theta -= alfa * gradient(theta, x, y)
        thetas[i, :] = theta.reshape(2)
    return theta, cost, thetas


def stochastic_gradient_descent(start_theta, x, y, alfa, iterations):
    theta = start_theta
    thetas = np.zeros((int(iterations/10), 2))
    cost = np.zeros((iterations*len(x), 1))
    for k in range(iterations):
        for i in range(len(x)):
            cost[k*len(x) + i] = 2 * cost_function(theta, x[i], y[i])
            theta -= alfa * gradient(theta, x[i], y[i])
            if i % 10 == 0:
                thetas[int(i/10), :] = theta.reshape(2)
    return theta, cost, thetas


def plot_contours(x, y, thetas, type):
    theta1 = np.linspace(-15, 15, 1000)
    theta2 = np.linspace(-15, 15, 1000)
    J = np.zeros((1000, 1000))
    for i, t1 in enumerate(theta1):
        for j, t2 in enumerate(theta2):
            J[i, j] = cost_function(np.array([[t1], [t2]]), x, y)

    X, Y = np.meshgrid(theta1, theta2)
    plt.figure()
    plt.contour(X, Y, J)
    plt.plot(thetas[:, 1], thetas[:, 0])
    plt.plot(thetas[-1, 1], thetas[-1, 0], 'bo')
    plt.title("Cost function and contour lines")
    plt.show()


def plot_cost_function_theta(x, y):
    theta1 = np.linspace(-15, 15, 1000)
    theta2 = np.linspace(-15, 15, 1000)
    J = np.zeros((1000, 1000))
    for i, t1 in enumerate(theta1):
        for j, t2 in enumerate(theta2):
            J[i, j] = cost_function(np.array([[t1], [t2]]), x, y)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(theta1, theta2)
    surf = ax.plot_surface(X, Y, J, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    high_limit = np.amax(J)
    low_limit = np.amin(J)
    ax.set_zlim(low_limit, high_limit)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_xlabel('theta2')
    ax.set_ylabel('theta1')
    plt.title("Cost function dependant on theta")
    plt.show()


def weighted_pseudo_inversion(x, y):
    high = np.max(x)
    low = np.min(x)
    m = len(x)
    ax = np.linspace(low, high, len(x))
    z = np.ones((m, 1))
    X = np.concatenate((z, x), axis=1)
    line = np.zeros((m, 1))

    for tau in np.linspace(0.5, 2, 4):
        w = np.zeros((m, m))
        for k in range(len(ax)):
            for i in range(len(x)):
                w[i, i] = np.exp(-(np.linalg.norm(ax[k] - x[i])**2)/(2*tau**2))
            inv = np.linalg.inv(np.dot(np.dot(X.T, w), X))
            theta = np.dot(inv, np.dot(np.dot(X.T, w), y))
            line[k] = np.dot(np.array([1, ax[k]]), theta)

        plt.figure(figsize=(8, 8))
        plt.plot(x, y, "ro")
        plt.plot(ax, line, "b-")
        plt.title("Regression line for locally \n weigthed linear regression for tau " + str(tau))
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

    return theta


def plot_cost_function(cost, type):

    plt.figure(figsize=(8, 8))
    plt.plot(cost[5:])
    plt.title("Cost as a function of number of \n iterations for " + type)
    plt.xlabel("Number of iterations")
    plt.ylabel("Value of cost J")
    plt.show()


if __name__ == "__main__":

    x, y = read_data()
    theta_pseudo = pseudo_inversion(x, y)

    theta_gradient, cost_gradient, thetas_gradient = gradient_descent(np.array([[1.], [1.]]), x, y, 0.0001, 5000)
    plot_cost_function(cost_gradient, "gradient descent")
    plot_contours(x, y, thetas_gradient, "gradijent")

    theta_stochastic, cost_stochastic, thetas_stochastic = stochastic_gradient_descent(np.array([[1.], [1.]]), x, y, 0.0001, 5000)
    plot_cost_function(cost_stochastic, "stochastic gradient descent")
    plot_contours(x, y, thetas_stochastic, "stochastic")

    weighted_pseudo_inversion(x, y)