import numpy as np
import matplotlib.pyplot as plt

def ShowOrignalImage(pixels):
    fig, axes = plt.subplots(6, 10, figsize=(11, 7), subplot_kw={'xticks':[], 'yticks':[]})
    for i, ax in enumerate(axes.flat):
        ax.imshow(np.array(pixels)[i].reshape(64, 64), cmap='gray')
        plt.show()
def Eigenfaces(pca):
    fig, axes = plt.subplots(3, 8, figsize=(9, 4), subplot_kw={'xticks':[], 'yticks':[]})
    for i, ax in enumerate(axes.flat):
        ax.imshow(pca.components_[i].reshape(64, 64), cmap='gray')
        ax.set_title("PC"+str(i+1))
    plt.show()

import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
df = pd.read_csv('FaceDataset.csv')
print(df)

x = df.drop(['TARGET'], axis=1)
y = df[['TARGET']]
pixels = df.drop(['TARGET'], axis=1)
print(np.array(pixels).shape)
ShowOrignalImage(pixels)

from sklearn.preprocessing import StandardScaler
x_std = StandardScaler().fit_transform(x)
x_std
features = x_std.T
covariance_matrix = np.cov(features)
print(covariance_matrix)

eig_vals, eig_vecs = np.linalg.eig(covariance_matrix)
print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)
eig_vals[0] / sum(eig_vals)

project_X = x_std.dot(eig_vecs.T[0])
project_X
result = pd.DataFrame(project_X, columns=['PC1'])
result['label'] = y

import matplotlib.pyplot as plt
import seaborn as sns
fig1 = plt.figure(1)
sns.regplot(x='label', y='PC1', data=result)
plt.title('PCA RESULT')
plt.show()

fig2 = plt.figure(2)
sns.regplot(x='label', y='')


