# https://www.datacamp.com/community/tutorials/principal-component-analysis-in-python
# https://www.askpython.com/python/examples/principal-component-analysis
# https://pub.towardsai.net/principal-component-analysis-pca-with-python-examples-tutorial-67a917bae9aa
# https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def my_PCA(input , n):
    standardized_input = input - np.mean(input , axis = 0)
    cov_matrix = np.cov(standardized_input , rowvar = False)
    eigen_values, eigen_vectors = np.linalg.eigh(cov_matrix)

    eigen_value_sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[eigen_value_sorted_index]
    sorted_eigenvectors = eigen_vectors[:,eigen_value_sorted_index]

    first_n_eigenvectors = sorted_eigenvectors[:, 0 : n]
    reduced_input = np.dot(first_n_eigenvectors.transpose(), standardized_input.transpose()).transpose()

    return reduced_input, sorted_eigenvalue, sorted_eigenvectors





# df = pd.read_csv('./A4/code/e3_breast_cancer.csv')
# df = df.iloc[:, : -1]
# label_column_name = 'diagnosis'
# label = df.pop(label_column_name)


df = pd.read_csv('./A4/code/e3_glass.csv')
label_column_name = 'Type'
label = df.pop(label_column_name)

print(df.head())
print(df.shape)


reduce_to_n = 2
column_names = ['PC'+str(i) for i in range(1, reduce_to_n+1)]
reduced_df, sorted_eigenvalue, sorted_eigenvectors = my_PCA(df , reduce_to_n)
reduced_df = pd.DataFrame(reduced_df, columns = column_names)
reduced_label_df= pd.concat([reduced_df , pd.DataFrame(label)] , axis = 1)

# # 3.2.
# print(sorted_eigenvalue)
# print(sorted_eigenvectors)
# print(f'<v_1, v_2> = {np.dot(sorted_eigenvectors[0], sorted_eigenvectors[1])}')
# print(f'<v_2, v_3> = {np.dot(sorted_eigenvectors[1], sorted_eigenvectors[2])}')
# print(f'<v_1, v_3> = {np.dot(sorted_eigenvectors[0], sorted_eigenvectors[2])}')



plt.figure()
sb.scatterplot(data = reduced_label_df, x = 'PC1', y = 'PC2', hue = label_column_name, s = 50, legend='full', palette= 'icefire')

# 3.3.
# pca = PCA().fit(df)
# plt.plot(np.cumsum(pca.explained_variance_ratio_), color='b', label = 'cumulative')
# plt.plot(pca.explained_variance_ratio_, color='r', label = 'individual')
# plt.xlabel('number of components')
# plt.xticks([i for i in range(df.shape[1])])
# # plt.xticks([i for i in range(df.shape[1], 2)])
# plt.legend()
# plt.ylabel('explained variance')

plt.show()