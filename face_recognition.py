import numpy as np
import numpy.linalg as npla
import numpy.matlib as npml
import cv2

def detect_faces(current_frame):
    current_frame_gray = cv2.cvtColor(current_frame, cv2.BGR2GRAY)
    faces = face_cascade.detectMultiScale(current_frame_gray, 1.3, 5)
    return faces

def store_faces(current_frame,counter):

    if len(faces) > 0:
        for (x,y,w,h) in faces:

            imshow('Captured Image',current_frame)

            face_crop_resize = crop_resize_faces(current_frame, np.array([x,y,w,h))
            # Display image and store in file
            cv2.imshow('Detected Face',face_crop)
            cv2.imshow('Resized Face',face_crop_resize)
            waitKey(0)

            filename = "face_%d.jpeg" % counter
            cv2.imwrite(filename, face_crop_resize, CV_IMWRITE_JPEG_QUALITY, 95)

            print "Image stored as 'face_%d.jpeg'" % counter
            counter++
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
    cv2.resize(face_crop,dsize,face_crop_resize)

    return face_crop_resize

def vectorize_image(A):
    size_vector_img = A.shape[0]*A.shape[1]
    mean_vector = np.zeros((size_vector_img))

    np.reshape(A, (size_vector_img,1))
    vector_img = A

    return vector_img

def image_classifier(y,trained_data):
    for i in range(3):
        y = np.tile(y,(20,1))
        diff = y - trained_data[:,:,i].T

        for j in range(20):
            diff_norm[j] = npla.norm(diff[j,:])

        mean_sqr_err[i] = np.mean(diff_norm)

    return np.argsort(mean_sqr_err)[2] + 1
