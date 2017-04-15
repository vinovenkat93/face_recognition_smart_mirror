import numpy as np
import numpy.linalg as npla
import face_recognition as face_reg
import cv2


# Load images from current folder
no_of_classes = 3
no_of_imgs_class = 50
max_prin_comp = 20
N = 150 # Number of images

# images stored here
train_image_path = "/home/sriranjitha/ECE_568_Project/train_images"

k = 0

# Vectorizing images and getting mean
for i in range(no_of_classes):
    for j in range(no_of_imgs_class):
        path = train_image_path + ("%d_%d.jpg") % (i,j)
        A = cv2.imread(path,0)
        A = cv2.cvtColor(A, cv2.BGR2GRAY)
        A.astype(double)

        vector_img[:,k] = face_reg.vectorize_image(A)

        mean_vector += vector_img
        k += 1

# Covariance matrix calculation
C = np.zeros((size_vector_img,size_vector_img))
#X = np.zeros(size_vector_img,1)

for i in range(N):
    norm_vector_img[:,i] = (vector_img[:,i] - mean_vector)/npla.norm((vector_img[:,1] - mean_vector))

# Getting first 20 eigenvectors
w,v = npla.eig(norm_vector_img)
index = np.argsort(w, kind="mergesort")

# Principal components
for i in range(max_prin_comp):
    W[:,i] = v[:,index[-i]]

# PCA of training images
k = 0
for i in range(no_of_classes):
    for j in range(no_of_imgs_class):
        y[:,j,i] = dot(W.transpose(),(vector_img[:,k] - mean_vector))
        k += 1

# Binary file
np.save('Trained_Data.npy',y)
np.save('PCA_matrix.npy',W)
np.save('mean_vector.npy',mean_vector)

# Text format
np.savetxt('Trained_Data.txt',y)
np.savetxt('PCA_matrix.txt',W)
np.savetxt('mean_vector.txt',mean_vector)
