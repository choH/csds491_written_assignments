from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

def plot_poisson(lam, x_max = 12):
    x = np.arange(0, x_max)
    y = stats.poisson.pmf(x, lam)
    plt.title(f"lambda = {lam}")
    plt.bar(x, y)
    plt.show()




# plot_poisson(2)
# plot_poisson(3)
# plot_poisson(4)

e = [0.53, 0.65, 0.91, 1.19, 1.30, 1.33, 1.90, 2.01, 2.48]
alpha, _, beta = stats.gamma.fit(e, floc=0)

def plot_gamma(alpha, beta, T):
    x = np.linspace(0, 9, 1000)
    n = np.arange(10)

    for t in T:
        y = stats.gamma.pdf(n, alpha, scale = beta + t)
        plt.plot(n, y, label = f"T = {t}")
    plt.legend()
    plt.show()


plot_gamma(alpha, beta, [0, 0.5, 1.5])