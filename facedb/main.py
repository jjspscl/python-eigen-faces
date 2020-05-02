from sklearn                  import decomposition
from sklearn                  import datasets
from sklearn.preprocessing    import StandardScaler
from PIL                      import Image
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os, sys, time

class EigenFace(object):
  
  def __init__(self, image_path):
    self.matrix = self.imgMatrix(image_path)
    self.group = image_path.split('\\')[1]
    self.index = image_path.split('\\')[-1].replace('.pgm', '')
    self.covariance_matrix = self.CVMatrix()
    self.eig_vals, self.eig_vecs = np.linalg.eig(self.covariance_matrix)

  def imgMatrix(self, path):
    infile = open(path, 'r', errors='ignore')
    header = infile.readlines()[1:3]
    max_val = int(header[1])
    width, height = [int(item) for item in header[0].split()]
    infile.seek(len(header))
    return np.array(Image.open(path)).reshape((height, width)) #.flatten()

  def FitTransform(self):
    return StandardScaler().fit_transform(self.matrix)

  def CVMatrix(self):
    features = self.matrix.T
    return np.cov(features)
    

# def leastCommon(data): # FOR PLOT COMPUTATION
#   min_rows, min_cols = sys.maxsize, sys.maxsize
#   max_rows, max_cols = 0, 0
#   dtype = type(data).__name__ 
#   if dtype == 'dict':
#     for k, v in data.items():
#       min_cols = min(min_cols, len(v))
#       max_cols = max(max_cols, len(v))
#     return min_cols, max_cols
#   elif dtype == 'list': print('ITS A LIST')

def ShowImages(data):
    plt.imshow(data.matrix, cmap='gray')
    plt.show()

def main():
  # LOOP THROUGH ALL DIRECTORIES
  faces = {}
  fig, axes = plt.subplots()
  for subdir, dirs, files in os.walk('.'):
    for file in files:
      if file.endswith('.pgm'):
        ef = EigenFace(os.path.join(subdir, file))
        group = ef.group

        #APPENDS DATA SET OBJECTS
        try:
          faces[ef.group].append(ef)
        except KeyError:
          faces.update({ef.group: [ef]})

  # ITERATE ALL DATASETS OBJECTS
  for k, v in faces.items(): 
    for eg in v: 
      ShowImages(eg)
      print('\nCOVARIANCE MATRIX: ', eg.covariance_matrix)
      print('\nEigenvectors:', eg.eig_vecs)
      print('\nEigenvalues:', eg.eig_vals)
      print('\n', eg.eig_vals[0] / sum(eg.eig_vals))

      project_X = eg.FitTransform().dot(eg.eig_vecs.T[0])
      print('\n', project_X)

if __name__ == "__main__":  # EXECUTION POINT
  os.system('cls')
  main()

# Phương Nguyễn
