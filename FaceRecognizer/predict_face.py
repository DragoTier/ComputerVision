import cv2
import training_data
import numpy as np

print("Loading face recognizer...")
face_recognizer = cv2.face_LBPHFaceRecognizer.create()
face_recognizer.read("./model/model.XML")

img = cv2.imread("./images/face_recognition/merkel_1.jpg")   # Enter Image name
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

label = face_recognizer.predict(gray)
subjects = training_data.get_subjects()
print("Guessing that it is " + subjects[label[0]] + " with " + str(label[1]) + "% probability.")
