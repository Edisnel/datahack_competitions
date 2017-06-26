__author__ = 'Edisnel C.C.'

import numpy as np
import pandas as pd
from sklearn.externals import joblib

from skimage.feature import hog

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import model_selection
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=affine_flags)
    return img

def convert(imgf, labelf, outf, n):
    f = open(imgf, "rb")
    l = open(labelf, "rb")
    o = open(outf, "w")
    f.read(16)
    l.read(8)
    images = []
    for i in range(1, n):
        image = [ord(l.read(1))]
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)
    for image in images:
        o.write(",".join(str(pix) for pix in image) + "\n")
    f.close()
    o.close()
    l.close()

#convert("Dataset/train-images", "Dataset/train-labels", "Dataset/mnist_train.csv", 60000)
#convert("Dataset/t10k-images", "Dataset/t10k-labels", "Dataset/mnist_test.csv", 10000)

# First column is the label
train = pd.read_csv("Dataset/mnist_train.csv")
test = pd.read_csv("Dataset/mnist_test.csv")

#dataset = datasets.fetch_mldata("MNIST Original")

features = np.array(train.iloc[:, 1:len(train.columns)], 'int16')
labels = np.array(train.iloc[:, 0], 'int')

test_features = np.array(test.iloc[:, 1:len(test.columns)], 'int16')
test_labels = np.array(test.iloc[:, 0], 'int')

std_scaler = StandardScaler()
list_hog_fd = []
for feature in features:
    fd = hog(feature.reshape((28,28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    fd = std_scaler.fit_transform(fd)
    list_hog_fd.append(fd)
hog_features = np.array(list_hog_fd, 'float64')

#-------------------------------- SVC ------------------------------------------
"""
#kernel = linear, poly, rbf, sigmoid -- C = 5?
param_test = {'gamma': np.logspace(-6, -1, 10)}
#'degree': (2, 3, 4)

svc = SVC(kernel='rbf')
clf = GridSearchCV(estimator=svc, param_grid=param_test, scoring='accuracy', verbose=True, cv=5)

# Antes del entrenamiento llevar a escala, estandarizar
std_scaler = StandardScaler()
hog_features = std_scaler.fit_transform(hog_features)

clf.fit(hog_features, labels)
joblib.dump(clf, "digitsSVMScale_cls.pkl", compress=3)
"""

# -------------------------------Neural Network -------------------------------------------------------------------

#param_test = {'alpha':(0.0001, 0.01, 0.001),'hidden_layer_sizes':((40,80),(40,28),(60,28), (70,30))}
param_test = {'hidden_layer_sizes': ((1080,700),(1000,1000))}

clf = GridSearchCV(estimator =\
       MLPClassifier(activation='relu', alpha=0.035938136638046257, batch_size='auto',\
       beta_1=0.1, beta_2=0.999, early_stopping=True,\
       epsilon=1e-08, learning_rate='adaptive', hidden_layer_sizes=(8192, 8192),\
       learning_rate_init=0.5, max_iter=1000, momentum=0.01,\
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,\
       solver='sgd', tol=0.0001, validation_fraction=0.1, verbose=True,\
       warm_start=False),\
       param_grid=param_test, scoring='accuracy',n_jobs=2, verbose=True, cv=5)

clf.fit(hog_features, labels)

#print gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
print clf.best_params_, "\n", clf.best_score_

joblib.dump(clf, "digits_neuralNet2.pkl", compress=3)



















