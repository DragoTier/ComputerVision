import cv2
import training_data
import numpy as np
import picamera
from picamera.array import PiRGBArray
import time

print("Starting camera...")

camera = picamera.PiCamera()
raw_capture = PiRGBArray(camera)

time.sleep(0.1)
camera.capture(raw_capture, format='bgr')

print("Image taken...")

print("Loading face recognizer...")
face_recognizer = cv2.face_LBPHFaceRecognizer.create()
face_recognizer.read("./model/model.XML")

# img = cv2.imread("./images/face_recognition/trump_1.jpg")   # Enter Image name
img = raw_capture.array

if img is not None:

    faces_and_gray_img = training_data.detect_face(img)

    if faces_and_gray_img is not None:
        faces, gray = faces_and_gray_img

        gray = cv2.equalizeHist(gray)  # equalize Histogram

        face = training_data.extract_face(gray, faces)

        training_data._show_faces(img, faces)

        label = face_recognizer.predict(face)
        subjects = training_data.get_subjects()

        if subjects is not None:

            if label[1] < 14.3:
                print("Guessing that it is " + subjects[label[0]] + " with a distance of " + str(label[1]) + ".")
            else:
                print("Unknown face detected. Distance of " + str(label[1]) + " was too high. Maybe it could be " + subjects[
                    label[0]] + ".")
        else:
            print("No subjects found!")
    else:
        print("NO face detected!")
else:
    print("No image found!")
