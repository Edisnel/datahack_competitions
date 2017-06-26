__author__ = 'Edisnel C.C.'

import cv2
from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np
from sklearn import metrics
import pandas as pd
from sklearn.preprocessing import StandardScaler

test = pd.read_csv("Dataset/mnist_test.csv")
test_features = np.array(test.iloc[:, 1:len(test.columns)], 'int16')
test_labels = np.array(test.iloc[:, 0], 'int')

def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(img,M,(SZ, SZ),flags=affine_flags)
    return img

#clf = joblib.load("digits_cls.pkl")
clf = joblib.load("digits_neuralNet2.pkl")

# Read the input image
ruta_img="/media/Datos/DataScience/Documentacion/Coding_Skills_Portafolio/Identify_Digits/images/d3.jpg"
im = cv2.imread(ruta_img)

# Convert to grayscale and apply Gaussian filtering
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

# Threshold the image
ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)

# Find contours in the image
ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get rectangles contains each contour
rects = [cv2.boundingRect(ctr) for ctr in ctrs]

std_scaler = StandardScaler()

#For each rectangular region, calculate HOG features and predict the digit using Linear SVM.
for rect in rects:
    cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 1)
    # Make the rectangular region around the digit
    leng = int(rect[3] * 0.8)
    pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
    pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
    roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
    # Resize the image
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    roi = cv2.dilate(roi, (3, 3))
    # Calculate the HOG features
    roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    # si en el entrenamiento se utilizo scaler, hacerlo aqui
    roi_hog_fd = std_scaler.fit_transform(roi_hog_fd)
    nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
    cv2.putText(im, str(int(nbr[0])), (rect[0], rect[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)

"""
list_hog_fd = []
for feature in test_features:
    fd = hog(feature.reshape((28,28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    list_hog_fd.append(fd)
hog_features = np.array(list_hog_fd, 'float64')

# Now predict the value of the digit on the second half:
predicted = clf.predict(hog_features)

print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(test_labels, predicted)))
"""

cv2.imshow("Resulting Image with Rectangular ROIs", im)
cv2.waitKey()











