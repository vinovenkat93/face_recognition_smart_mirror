import numpy as np
import numpy.linalg as npla
import face_recognition as face_reg
import cv2

def run_pca():
    global user
    capture_obj = cv2.VideoCapture(0) #Index: 0 (Camera 1)
    face_cascade = cv2.CascadeClassifier('')

    if not capture_obj.isOpened():
        capture_obj.open()

        trained_data = np.load('Trained_Data.npy')
        PCA_matrix = np.load('PCA_matrix.npy')
        mean_vector = np.load('mean_vector.npy')

        # Use interrupt from a Motion detector instead of constant polling
        while True:

            isread, current_frame = capture_obj.read()
            if not isread:
                continue
            else:
                # K-nearest neighbor for finding right face
                faces = face_reg.detect_faces(face_cascade,current_frame)
                for (x,y,w,h) in faces:

                    face_crop_resize = face_reg.crop_resize_faces(current_frame, np.array([x,y,w,h]))
                    vector_img = face_reg.vectorize_image(face_crop_resize)
                    vector_img = (vector_img - mean_vector)/npla.norm(vector_img - mean_vector)
                    y_face = np.dot(PCA_matrix.T,(vector_img - mean_vector))

                    img_class = face_reg.image_classifier(y_face,trained_data)

                    if img_class == 1:
                        user = "Vinoth"
                    elif img_class == 2:
                        user = "Varshini"
                    else:
                        user = "Ridhi"
