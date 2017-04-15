# This module is used to collect training images (Faces of users who will be using the mirror)

import numpy as np
import cv2
import face_recognition as face_reg

capture_obj = cv2.VideoCapture(0) #Index: 0 (Camera 1)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier('')
counter = 1

while True:

    if not capture_obj.isOpened():
        capture_obj.open()

    input_text = raw_input("Press Enter if you want to capture image")

    if input_text == "":
        #Capture frame
        isread, current_frame = capture_obj.read()

        if not isread:
            print "Image couldn't be captured. Please try again!"
            continue

        faces = face_reg.detect_faces(face_cascade,current_frame)
        if not (face_reg.store_faces(current_frame,faces)):
            print "Couldn't detect faces. Try again!"

    else:
        print "You typed something else. Press enter to capture and store image"
