import cv2
import training_data
import numpy as np
import picamera
from picamera.array import PiRGBArray
import time

print("Starting camera...")

LIGHT_RED = '\033[91m'
LIGHT_GREEN = '\033[32m'
END = '\033[0m'

raw_capture = None

with picamera.PiCamera() as camera:

    raw_capture = PiRGBArray(camera)

    time.sleep(0.1)
    camera.capture(raw_capture, format='bgr')

print("Image taken...")

print("Loading face recognizer...")
face_recognizer = cv2.face_LBPHFaceRecognizer.create()


# SET MODEL CONFIG
training_data.set_model_configuration(training_data.ModelConfiguration.AllWithEqualization)
model_file = training_data.get_model_name()

if model_file is not None:

    face_recognizer.read("./model/" + model_file)

else:

    face_recognizer.read("./model/model.XML")

# img = cv2.imread("./images/face_recognition/trump_1.jpg")   # Enter Image name

if raw_capture is not None:

    img = raw_capture.array

    faces_and_gray_img = training_data.detect_face(img)

    if faces_and_gray_img is not None:
        faces, gray = faces_and_gray_img

        gray = cv2.equalizeHist(gray)  # equalize Histogram

        face = training_data.extract_face(gray, faces)

        label = face_recognizer.predict(face)
        subjects = training_data.get_subjects_from_json()

        if subjects is not None:

            if label[1] < 14.3:
                print("Guessing that it is " + LIGHT_GREEN + subjects[str(label[0])] + END + " with a distance of " +
                      LIGHT_GREEN + str(label[1]) + END + ".")
            else:
                print(LIGHT_RED + "Unknown face detected. " + END + "Distance of " + LIGHT_RED + str(label[1]) + END +
                      " was too high. Maybe it could be " + LIGHT_RED + subjects[str(label[0])] + END + ".")

            training_data._show_faces(img, faces)

        else:
            print("No subjects found!")

    else:
        print("No face detected!")
else:
    print("No image found!")
