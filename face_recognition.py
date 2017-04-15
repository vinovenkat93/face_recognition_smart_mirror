import numpy as np
import cv2

def detect_faces(current_frame):
    current_frame_gray = cv2.cvtColor(current_frame, cv2.BGR2GRAY)
    faces = face_cascade.detectMultiScale(current_frame_gray, 1.3, 5)
    return faces

def store_faces(counter):

    if len(faces) > 0:
        for (x,y,w,h) in faces:

            imshow('Captured Image',curren_frame)
            face_crop = curren_frame[y:y+h, x:x+w]
            dsize = (50,50) # resized_image_size

            # Display image and store in file
            cv2.imshow('Detected Face',face_crop)
            cv2.resize(face_crop,dsize,face_crop_resize)
            cv2.imshow('Resized Face',face_crop_resize)
            waitKey(0)

            filename = "face_%d.jpeg" % counter
            cv2.imwrite(filename, face_crop_resize, CV_IMWRITE_JPEG_QUALITY, 95)

            print "Image stored as 'face_%d.jpeg'" % counter
            counter++
            return True
    else:
        return False

def
