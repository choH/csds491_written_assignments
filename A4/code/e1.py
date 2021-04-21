# Ref. https://matplotlib.org/3.2.2/gallery/statistics/confidence_ellipse.html#sphx-glr-gallery-statistics-confidence-ellipse-py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
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

    cov = np.cov(x, y)
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
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def plot_multivariate_normal(mu, cov_matrix, size = 500, show_eigen_vector = False, show_sigma_contour = False):
    plt.axis('equal')
    ax = plt.gca()
    x, y = np.random.multivariate_normal(mu, cov_matrix, size).T
    ax.scatter(x,y)

    if show_eigen_vector:
        eigen_value, eigen_vector = np.linalg.eig(cov_matrix)
        ax.quiver([mu[0], mu[0]], [mu[1], mu[1]], eigen_vector[0, :], eigen_vector[1, :], color=['r','b'], scale=4)

    if show_sigma_contour:
        confidence_ellipse(x, y, ax, n_std=1,
            label=r'$1\sigma$', edgecolor='darkslategray')
        confidence_ellipse(x, y, ax, n_std=2,
            label=r'$2\sigma$', edgecolor='darkcyan', linestyle='--')
        confidence_ellipse(x, y, ax, n_std=3,
            label=r'$3\sigma$', edgecolor='darkturquoise', linestyle=':')
        ax.legend()

    plt.show()

cov_mat_uncorrelated = np.array([[1, 0], [0, 1]])
cov_mat_correlated = np.array([[1, 0.9], [0.9, 1]])
cov_mat_anticorrelated = np.array([[1, -0.9], [-0.9, 1]])

mu_1, mu_2, mu_3 = np.array([0, 0]), np.array([0.5, 0.5]), np.array([0.7, 0.7])

# 1.1.
# plot_multivariate_normal(mu_1, cov_mat_uncorrelated)
# plot_multivariate_normal(mu_2, cov_mat_correlated)
# plot_multivariate_normal(mu_3, cov_mat_anticorrelated)

# 1.2.
plot_multivariate_normal(mu_1, cov_mat_uncorrelated, show_eigen_vector = True, show_sigma_contour = True)
plot_multivariate_normal(mu_2, cov_mat_correlated, show_eigen_vector = True, show_sigma_contour = True)
plot_multivariate_normal(mu_3, cov_mat_anticorrelated, show_eigen_vector = True, show_sigma_contour = True)
