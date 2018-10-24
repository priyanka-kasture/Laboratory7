# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 20:55:08 2018

@author: Priyanka Kasture
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn import svm

x = [1, 5, 1.5, 8, 1, 9]
y = [2, 8, 1.8, 8, 0.6, 11]

plt.scatter(x,y)
plt.show()

# Feature list : X
X = np.array([[1,2],[5,8],[1.5,1.8],[8,8],[1,0.6],[9,11]])

# Target Labels
y = [0,1,0,1,0,1]

# Training the Classifier
# SVC is the function of SVM class, which in-turn is a class of the Sk-Learn Library.
classifier = svm.SVC(kernel='linear', C = 1.0)

# Fitting the model
classifier.fit(X,y)

# Testing
print(classifier.predict([[0.58,0.76]])) # Class 0
print(classifier.predict([[10.58,10.76]])) # Class 1

# Thus classified with 100% accuracy

w = classifier.coef_[0]
print(w)

'''
Weights assigned to the features (coefficients in the primal problem). 
This is only available in the case of a linear kernel.
'''
# Visualizing the Hyper-line

a = -w[0]/w[1]

xx = np.linspace(0,12)
yy = a*xx - classifier.intercept_[0]/w[1]

h = plt.plot(xx, yy, 'k-', label="Non-Weighted Division")

plt.scatter(X[:, 0], X[:, 1], c = y)
plt.legend()
plt.show()
