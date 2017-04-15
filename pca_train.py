import numpy as np
import cv2


# Load images from current folder
no_of_classes = 3
no_of_imgs_class = 50
max_prin_comp = 20
N = 150 # Number of images

# images stored here
train_image_path = "/home/sriranjitha/ECE_568_Project/train_images"

k = 0
a = 0

# Vectorizing images and getting mean
for i in range(no_of_classes):
    for j in range(no_of_imgs_class):
        path = train_image_path + ("%d_%d.jpg") % (i,j)
        A = cv2.imread(path,0)
        A = cv2.cvtColor(A, cv2.BGR2GRAY)
        A.astype(double)

        if a == 1:
            size_vector_img = A.shape[0]*A.shape[1]
            mean_vector = np.zeros((size_vector_img))
            a++

        np.reshape(A, (size_vector_img,1))
        vector_img[:,k] = A

        mean_vector += vector_img
        k++

# Covariance matrix calculation
C = np.zeros((size_vector_img,size_vector_img))
#X = np.zeros(size_vector_img,1)

for i in range(N):
    norm_vector_img[:,i] = (vector_img[:,i] - mean_vector)/np.linalg.norm((vector_img[:,1] - mean_vector))

# Getting first 20 eigenvectors
w,v = np.linalg.eig(norm_vector_img)
index = np.argsort(w, kind="mergesort")

# Principal components
for i in range(max_prin_comp):
    W[:,i] = v[:,index[-i]]

# PCA of training images
k = 0
for i in range(no_of_classes):
    for j in range(no_of_imgs_class):
        y[:,j,i] = dot(W.transpose(),(vector_img[:,k] - mean_vector))
        k++

# Binary file
np.save('Trained_Data.npy',y)

# Text format
np.savetxt('Trained_Data.txt',y)