import numpy as np
import kernels as k
import cvxopt
from explore_data import read_data, make_meshgrid, plot_line, visualize_data


class SVM:
    def __init__(self, kernel=k.linear_kernel):
        self.parameters = {"kernel": kernel}
        self.kernel = kernel
        self.C = None

    def train(self, x, y, C=None):
        m = x.shape[0]
        self.C = C

        K = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                K[i, j] = self.kernel(x[i], x[j])
        print(K)
        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(-np.ones((m, 1)))
        A = cvxopt.matrix(y, (1, m))
        b = cvxopt.matrix(0.0)
        if C is None:
            G = cvxopt.matrix(np.diag(-np.ones(m)))
            h = cvxopt.matrix(np.zeros(m))
        else:
            first_G = np.diag(-np.ones(m))
            second_G = np.diag(np.ones(m))
            G = cvxopt.matrix(np.vstack((first_G, second_G)))
            first_h = np.zeros(m)
            second_h = np.ones(m) * C
            h = cvxopt.matrix(np.hstack((first_h, second_h)))

        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        alpha = np.ravel(solution['x'])

        support_vectors = alpha > 1e-5
        alpha = alpha[support_vectors]
        alpha = alpha.reshape(len(alpha), 1)
        support_vectors_x = x[support_vectors, :]
        support_vectors_y = y[support_vectors]

        w = np.sum(alpha * support_vectors_y * support_vectors_x, axis=0).reshape(2, 1)
        m1 = np.argmin(self.kernel(support_vectors_x[np.where(support_vectors_y == 1)[0], :], w.T))
        m2 = np.argmax(self.kernel(support_vectors_x[np.where(support_vectors_y == -1)[0], :], w.T))
        bias = -(self.kernel(support_vectors_x[m1, :].reshape(1, 2), w.T) + self.kernel(support_vectors_x[m2, :].reshape(1, 2), w.T))/2

        self.parameters["w"] = w
        self.parameters["b"] = bias
        self.parameters["support_vectors"] = [support_vectors_x, support_vectors_y]
        self.parameters["alpha"] = alpha
        return w, bias

    def predict(self, x):

        w = self.parameters["w"]
        b = self.parameters["b"]
        alpha = self.parameters["alpha"]
        support_vectors = self.parameters["support_vectors"]

        p = np.zeros((len(x), 1))
        for i in range(len(x)):
            for alpha_i, x_i, y_i in zip(alpha, support_vectors[0], support_vectors[1]):
                p[i] += alpha_i * y_i * self.kernel(x[i, :], x_i.reshape(1, len(x_i)))

        predictions = np.sign(p + b)
        return predictions

    def accuracy(self, prediction, y):
        return np.sum(prediction == y)/len(y)


if __name__ == "__main__":
    x, y = read_data("svmData_ls.txt")
    visualize_data(x, y)
    svm = SVM(kernel=k.linear_kernel)
    w, b = svm.train(x, y, 1)
    p = svm.predict(x)
    print(svm.accuracy(p, y))
    x0, x1 = make_meshgrid(x[:, 0], x[:, 1])
    X = np.c_[x0.ravel(), x1.ravel()]
    predictions = svm.predict(X)
    plot_line(x, y, svm.parameters, predictions)

    x, y = read_data("svmData_nls.txt")
    visualize_data(x, y)
    svm = SVM(kernel=k.polynomial_kernel)
    w, b = svm.train(x, y, 1)
    p = svm.predict(x)
    print(svm.accuracy(p, y))
    x0, x1 = make_meshgrid(x[:, 0], x[:, 1])
    X = np.c_[x0.ravel(), x1.ravel()]
    predictions = svm.predict(X)
    plot_line(x, y, svm.parameters, predictions)

