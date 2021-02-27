from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

def p_theta__beta(theta, y, n, alpha, beta):
    return stats.beta.pdf(theta, alpha + y, beta + n - y)


def plot_p_theta__beta(alpha, beta):
    x = np.linspace(0, 1, 100)
    y = stats.beta.pdf(x, alpha, beta)

    plt.xlabel("theta")
    plt.ylabel("p(theta)")
    plt.title(f"alpha = {alpha}, beta = {beta}")
    plt.plot(x, y)
    plt.show()


# plot_p_theta__beta(8, 2)
# plot_p_theta__beta(2, 8)
# plot_p_theta__beta(0.1, 0.1)
# plot_p_theta__beta(5, 5)
# plot_p_theta__beta(1, 1)


trial_list = [0, 1, 2, 5, 10, 100]

def plot_p_theta__beta_bernoulli(alpha, beta, theta, trial_list):
    f = stats.bernoulli(theta)
    data = f.rvs(1000)
    x = np.linspace(0, 1, 100)

    for i in range(len(trial_list)):
        n = trial_list[i]
        y = sum(data[:n])
        posterior = p_theta__beta(x, y, n, alpha, beta)
        plt.plot(x, posterior, label = f'n = {n}')
    plt.legend()
    plt.xlabel("p(theta | y, n)")
    plt.ylabel("theta")
    plt.title(f"alpha = {alpha}, beta = {beta}")

    plt.show()

# plot_p_theta__beta_bernoulli(2, 8, 0.5, trial_list)
# plot_p_theta__beta_bernoulli(8, 2, 0.5, trial_list)
# plot_p_theta__beta_bernoulli(0.1, 0.1, 0.5, trial_list)
# plot_p_theta__beta_bernoulli(5, 5, 0.5, trial_list)
# plot_p_theta__beta_bernoulli(1, 1, 0.5, trial_list)

