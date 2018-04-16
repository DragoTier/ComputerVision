import cv2
import training_data
import numpy as np

print("Loading face recognizer...")
face_recognizer = cv2.face_LBPHFaceRecognizer.create()
face_recognizer.read("./model/model.XML")

img = cv2.imread("./images/face_recognition/ivo_1.jpg")   # Enter Image name

faces_and_gray_img = training_data.detect_face(img)

if faces_and_gray_img is not None:

    faces, gray = faces_and_gray_img

    face = training_data.extract_face(gray, faces)

    training_data._show_faces(img, faces)

    label = face_recognizer.predict(face)
    subjects = training_data.get_subjects()

    print(label)

    # cv2.imshow("Shooting", face)
    # cv2.waitKey(0)

    if (label[1] < 80):
        print("Guessing that it is " + subjects[label[0]] + " with a distance of " + str(label[1]) + ".")
    else:
        print("Unknown face detected. Distance of " + str(label[1]) + " was too high.")
else:
    print("NO face detected!")
