import cv2
import numpy as np
import matplotlib.image as cv
import matplotlib.pyplot as plt
import copy

def bw_image(path = 'A3/code/hello_world.png'):
    img = cv.imread(path)
    img = img.copy()
    img[img < 0.5] = -1
    img[img >= 0.5] = 1
    img = img[: , :, 0]
    return img


def add_noise(img, thlr = 0.95):
    m, n = np.shape(img)
    noised = np.copy(img)
    for i in range(m):
        for j in range(n):
            if np.random.rand() >= thlr:
                noised[i][j] = -noised[i][j]
    return noised



def neighbor_sum(x, i, j, more_flag = False):
    def get_pix(x, i, j):
        try:
            return x[i][j]
        except IndexError:
            return 0
    if not more_flag:
        sum = x[i][j] * (get_pix(x, i+1, j) + get_pix(x, i-1, j) + get_pix(x, i, j+1) + get_pix(x, i, j-1))
    else:
        sum = x[i][j] * (get_pix(x, i+1, j) + get_pix(x, i-1, j) + get_pix(x, i, j+1) + get_pix(x, i, j-1)
                    + get_pix(x, i+1, j+1) + get_pix(x, i-1, j-1) + get_pix(x, i-1, j+1) + get_pix(x, i+1, j-1)
                    + get_pix(x, i+2, j) + get_pix(x, i-2, j) + get_pix(x, i, j+2) + get_pix(x, i, j-2))
    return sum


def denoise(img, h=5, beta=5, eta=5, energy_plot = False, energy_cap = False, neighbor_eval = False):
    m, n = np.shape(img)
    x = np.copy(img)
    y = np.copy(img)

    pass_counter = 0
    energy_change = []
    current_energy = 0
    if energy_plot:
        current_energy = init_energy(img, h, beta, eta)
        energy_change.append(current_energy)


    for i in range(m):
        for j in range(n):
            x_k = x[i][j]
            first_term = -2 * h * x_k
            second_term = 4 * beta * neighbor_sum(x, i, j)
            third_term = 2 * eta * x_k * y[i][j]
            neighbor_term = 4 * beta * neighbor_sum(x, i, j, more_flag = True)
            if not neighbor_eval:
                delta = first_term + second_term + third_term
            else:
                delta = first_term + neighbor_term + third_term
            if delta < 0:
                x[i][j] = -x[i][j]
                current_energy += delta
            pass_counter += 1
            energy_change.append(current_energy)
            if energy_cap:
                if pass_counter > m * n // 2:
                    break
    if energy_plot:
        plt.plot([i for i in range(pass_counter+1)], energy_change)
        plt.xlabel("Number of Passes")
        plt.ylabel("Total Energy")
        plt.show()
    return x

def init_energy(img, h=5, beta=5, eta=5):
    m, n = np.shape(img)
    x = np.copy(img)
    y = np.copy(img)

    init_energy = 0
    for i in range(m):
        for j in range(n):
            x_k = x[i][j]
            first_term = h * x_k
            second_term = -beta * neighbor_sum(x, i, j)
            third_term = -eta * x_k * y[i][j]
            delta = first_term + second_term + third_term
            init_energy += delta
    return init_energy


# img = bw_image('A3/code/quin.png') # must be png, jpg might break due to write_flag.
img = bw_image('A3/code/zebra_dia.png') # must be png, jpg might break due to write_flag.
# img = bw_image('A3/code/hellow_word_bw.png') # must be png, jpg might break due to write_flag.
# plt.imshow(img, cmap = 'gray')
# plt.show()

noised_img = add_noise(img)
# plt.imshow(noised_img, cmap = 'gray')
# plt.show()

# denoised_img = denoise(noised_img)
# denoised_img = denoise(noised_img, energy_plot = True)
# denoised_img = denoise(noised_img, energy_plot = False, energy_cap = True)
# denoised_img = denoise(noised_img, 100, 5, 5)
# denoised_img = denoise(noised_img, 0, 5, 5) #E1.5.1

denoised_img = denoise(noised_img, 0, 2, 2) #1.5.2 # print(np.asarray(denoised_img))
plt.imshow(denoised_img, cmap = 'gray')
plt.show()
# denoised_img = denoise(noised_img, 0, 0.1, 1, neighbor_eval = True) #1.5.2 # print(np.asarray(denoised_img))
# plt.imshow(denoised_img, cmap = 'gray')
# plt.show()