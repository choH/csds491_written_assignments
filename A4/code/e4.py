# Ref. https://towardsdatascience.com/gaussian-mixture-modelling-gmm-833c88587c7f
# https://matplotlib.org/3.2.2/gallery/statistics/confidence_ellipse.html#sphx-glr-gallery-statistics-confidence-ellipse-py
# http://ethen8181.github.io/machine-learning/clustering/GMM/GMM.html#Parameter-Estimation


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse
from sklearn import datasets
import matplotlib.transforms as transforms

np.random.seed(0)

def confidence_ellipse(x, y, ax, n_std=3.0, mean=[], cov=[], facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y) if len(cov) == 0 else cov
    mean_x = np.mean(x) if len(mean) == 0 else mean[0]
    mean_y = np.mean(y) if len(mean) == 0 else mean[1]

    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def plot_sigma_contour(data, means, covs, title = 'old_faithful'):
    # plt.plot(data[:, 0], data[:, 1], '.', color="steelblue")
    # plt.plot(data[:, 0], data[:, 1], '.', color="steelblue", label = 'Species')
    # versicolor = data[data['species']=='versicolor']
    # virginica = data[data['species']=='virginica']
    # plt.plot(versicolor["petal_length"], versicolor["petal_width"], '*',label="versicolor", color = 'orange')
    # plt.plot(virginica["petal_length"], virginica["petal_width"], '+',label="virginica", color = 'blue')
    # plt.axis('equal')
    ax = plt.gca()

    x = np.arange(-1, 6, 0.001)
    y = np.arange(-1, 6, 0.001)

    for i in range(means.shape[0]):
        mean = means[i]
        cov = covs[i]
        confidence_ellipse(x, y, ax, n_std=1, mean=mean, cov=cov,
            label=r'$1\sigma$', edgecolor='darkslategray')
        confidence_ellipse(x, y, ax, n_std=2, mean=mean, cov=cov,
            label=r'$2\sigma$', edgecolor='darkcyan', linestyle='--')
        confidence_ellipse(x, y, ax, n_std=3, mean=mean, cov=cov,
            label=r'$3\sigma$', edgecolor='darkturquoise', linestyle=':')
    ax.legend()
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    # plt.title("$1, 2, 3\sigma$ contours on Old Faithful Dataset")
    # plt.title("$1, 2, 3\sigma$ contours on IRIS Dataset")

    plt.title(title)

    plt.show()

# http://ethen8181.github.io/machine-learning/clustering/GMM/GMM.html#Parameter-Estimation
class GMM:

    def __init__(self, k_cluster, n_iter = 10, stop_thrl = 0.000001):
        self.k_cluster = k_cluster
        self.n_iters = n_iter
        self.stop_thrl = stop_thrl

    def fit(self, X):

        # data's dimensionality and member_matonsibility vector
        n_row, n_col = X.shape
        self.member_mat = np.zeros((n_row, self.k_cluster))

        # initialize parameters
        chosen = np.random.choice(n_row, self.k_cluster, replace = False)
        self.means = X[chosen]
        self.pi = np.full(self.k_cluster, 1 / self.k_cluster)

        shape = self.k_cluster, n_col, n_col
        self.cov = np.full(shape, np.cov(X, rowvar = False))

        log_likelihood = 0
        self.stop_flag = False
        self.log_likelihood_trace = []

        for i in range(self.n_iters):
            log_likelihood_new = self._do_estep(X)
            self._do_mstep(X)

            if abs(log_likelihood_new - log_likelihood) <= self.stop_thrl:
                self.stop_flag = True
                break

            log_likelihood = log_likelihood_new
            self.log_likelihood_trace.append(log_likelihood)

        return self

    def _do_estep(self, X):
        def compute_log_likelihood(X):
            for i in range(self.k_cluster):
                prior = self.pi[i]
                likelihood = multivariate_normal(self.means[i], self.cov[i]).pdf(X)
                self.member_mat[:, i] = prior * likelihood

        compute_log_likelihood(X)
        log_likelihood = np.sum(np.log(np.sum(self.member_mat, axis = 1)))

        # normalize
        self.member_mat = self.member_mat / self.member_mat.sum(axis = 1, keepdims = True) # https://stackoverflow.com/questions/39441517/in-numpy-sum-there-is-parameter-called-keepdims-what-does-it-do
        return log_likelihood


    def _do_mstep(self, X):
        member_mat_pi = self.member_mat.sum(axis = 0)
        # pi
        self.pi = member_mat_pi / X.shape[0]
        # mu
        weighted_sum = np.dot(self.member_mat.transpose(), X)
        self.means = weighted_sum / member_mat_pi.reshape(-1, 1)
        # cov
        for k in range(self.k_cluster):
            diff = (X - self.means[k]).transpose()
            weighted_sum = np.dot(self.member_mat[:, k] * diff, diff.transpose())
            self.cov[k] = weighted_sum / member_mat_pi[k]

        return self


data = np.loadtxt('./A4/code/faithful.txt')
# iris = datasets.load_iris()
# data = iris.data[:,2:4]

# model = GMM(k_cluster = 3, n_iter = 0, stop_thrl = 0.00001) # before fit

k_cluster_input = 3
n_iter_input = 3
model = GMM(k_cluster = k_cluster_input, n_iter = k_cluster_input, stop_thrl = 0.00001)
model.fit(data)

title = f'k = {k_cluster_input}, n_iter = {n_iter_input} on Old Faithful'


plot_sigma_contour(data, model.means, model.cov, title = title)

print(f'mean: {model.means}')
print(f'cov: {model.cov}')
print(f'pi: {model.pi}')

