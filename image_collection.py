import numpy as np
import cv2
import face_recognition as face_reg

capture_obj = cv2.VideoCapture(0) #Index: 0 (Camera 1)
face_cascade = cv2.CascadeClassifier('')
#eye_cascade = cv2.CascadeClassifier('')
counter = 1

while True:

    if !capture_obj.isOpened():
        capture_obj.open()

    input_text = raw_input("Press Enter if you want to capture image")

    if text == "":
        #Capture frame
        isread, current_frame = capture_obj.read()

        if !isread:
            print "Image couldn't be captured. Please try again!"
            continue

        faces = face_reg.detect_faces(current_frame)
        face_reg.store_faces(faces)

    else:
        print "You typed something else. Press enter to capture and store image"
