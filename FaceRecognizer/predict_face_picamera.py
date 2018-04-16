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

print(raw_capture.array)

print("Loading face recognizer...")
face_recognizer = cv2.face_LBPHFaceRecognizer.create()
face_recognizer.read("./model/model.XML")

# img = cv2.imread("./images/face_recognition/trump_1.jpg")   # Enter Image name
img = raw_capture.array
face = training_data.detect_face(img)

label = face_recognizer.predict(face)
subjects = training_data.get_subjects()

if(label[1] < 80):
    print("Guessing that it is " + subjects[label[0]] + " with a distance of " + str(label[1]) + ".")
else:
    print("Unknown face detected. Distance of " + str(label[1]) + " was too high.")

cv2.imshow("Shooting", raw_capture.array)
cv2.waitKey(0)
