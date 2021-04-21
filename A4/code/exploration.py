# https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b

import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
# from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# df = pd.read_csv('./A4/code/e3_breast_cancer.csv')
# df = df.iloc[:, : -1]
# label_column_name = 'diagnosis'
# label = df.pop(label_column_name)

df = pd.read_csv('./A4/code/e3_glass.csv')
label_column_name = 'Type'
label = df.pop(label_column_name)

print(df.head())
print(df.shape)

tsne = TSNE(n_components=2, verbose=1, perplexity=100, n_iter=300)
tsne_results = tsne.fit_transform(df)
reduce_to_n = 2
column_names = ['T-SNE '+str(i) for i in range(1, reduce_to_n+1)]
reduced_df = pd.DataFrame(tsne_results, columns = column_names)
reduced_label_df = pd.concat([reduced_df , pd.DataFrame(label)] , axis = 1)

# df['T-SNE 1'] = tsne_results[:,0]
# df['T-SNE 2'] = tsne_results[:,1]


plt.figure()
sb.scatterplot(
    x="T-SNE 1", y="T-SNE 2",
    hue=label_column_name,
    palette='icefire',
    data=reduced_label_df,
    legend="full",
    alpha=0.9
)
plt.show()