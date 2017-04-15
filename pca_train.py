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
a = 0

# Vectorizing images and getting mean
for i in range(no_of_classes):
    for j in range(no_of_imgs_class):
        path = train_image_path + ("%d_%d.jpg") % (i,j)
        A = cv2.imread(path,0)
        A = cv2.cvtColor(A, cv2.BGR2GRAY)
        A.astype(float)
        if a == 0:
#            vector_img = np.zeros((A.shape[0]*A.shape[1]))
            vector_img = face_reg.vectorize_image(A)
            mean_vector = vector_img
            a = 1
        else:
            vector_img = np.concatenate((vector_img,face_reg.vectorize_image(A)),1)
            mean_vector += vector_img[:,k]
        k += 1

mean_vector = mean_vector / N
# Covariance matrix calculation
#X = np.zeros(size_vector_img,1)
size_vector_img = A.shape[0]*A.shape[1]
norm_vector_img = np.zeros((size_vector_img,1))
norm_vector_img = (vector_img[:,0] - mean_vector)/npla.norm((vector_img[:,0] - mean_vector))

for i in range(N-1):
    norm_vector_img = np.concatenate((norm_vector_img,(vector_img[:,i+1] - mean_vector)/npla.norm(vector_img[:,i+1] - mean_vector)),1)

# Getting first 20 eigenvectors
w,v = npla.eig(norm_vector_img)
index = np.argsort(w, kind="mergesort")

W = v[:,index[-1]]
# Principal components
for i in range(max_prin_comp-1):
    W[:,np.newaxis] = v[:,index[-(i+1)]]

# PCA of training images
k = 0
y = np.zeros((max_prin_comp,no_of_imgs_class,no_of_classes))
for i in range(no_of_classes):
    for j in range(no_of_imgs_class):
        y[:,j,i] = np.dot(W.transpose(),(vector_img[:,k] - mean_vector))
        k += 1

# Binary file
np.save('Trained_Data.npy',y)
np.save('PCA_matrix.npy',W)
np.save('mean_vector.npy',mean_vector)

# Text format
np.savetxt('Trained_Data.txt',y)
np.savetxt('PCA_matrix.txt',W)
np.savetxt('mean_vector.txt',mean_vector)
