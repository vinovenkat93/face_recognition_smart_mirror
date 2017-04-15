# Contains all the helper functions needed for face recognition

import numpy as np
import numpy.linalg as npla
import numpy.matlib as npml
import cv2

def detect_faces(face_cascade, current_frame):
    current_frame_gray = cv2.cvtColor(current_frame, cv2.BGR2GRAY)
    faces = face_cascade.detectMultiScale(current_frame_gray, 1.3, 5)
    return faces

def store_faces(current_frame,faces):
    
    if len(faces) > 0:
        counter = 0
        for (x,y,w,h) in faces:

            cv2.imshow('Captured Image',current_frame)

            face_crop, face_crop_resize = crop_resize_faces(current_frame, np.array([x,y,w,h]))
            # Display image and store in file
            cv2.imshow('Detected Face',face_crop)
            cv2.imshow('Resized Face',face_crop_resize)
            cv2.waitKey(0)

            filename = "face_%d.jpeg" % counter
            cv2.imwrite(filename, face_crop_resize, cv2.CV_IMWRITE_JPEG_QUALITY, 95)

            print "Image stored as 'face_%d.jpeg'" % counter
            counter += 1
            return True
    else:
        return False

def crop_resize_faces(image, dimensions):
    x = dimensions[0]
    y = dimensions[1]
    w = dimensions[2]
    h = dimensions[3]

    face_crop = image[y:y+h, x:x+w]
    dsize = (50,50) # resized_image_size
    face_crop_resize = np.zeros(dsize)
    cv2.resize(face_crop,dsize,face_crop_resize)

    return face_crop, face_crop_resize

def vectorize_image(A):
    size_vector_img = A.shape[0]*A.shape[1]
    vector_img = np.reshape(A, (size_vector_img,1)) 
    return vector_img

def image_classifier(y,trained_data):
    mean_sqr_err = np.zeros((3,1))
    for i in range(3):
        y = np.tile(y,(20,1))
        diff = y - trained_data[:,:,i].T
    
        diff_norm = np.zeros((20,1))
        for j in range(20):
            diff_norm[j] = npla.norm(diff[j,:])

        mean_sqr_err[i] = np.mean(diff_norm)

    return np.argsort(mean_sqr_err)[2] + 1
