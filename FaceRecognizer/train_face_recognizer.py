import cv2
import training_data
import numpy as np

# https://www.superdatascience.com/opencv-face-recognition/

print("Starting with opencv version " + cv2.__version__)

# https://docs.opencv.org/3.0-beta/modules/face/doc/facerec/facerec_api.html#createlbphfacerecognizer
face_recognizer = cv2.face_LBPHFaceRecognizer.create()

images, labels, subjects = training_data.get_training_data()

images_np = np.array(images)
labels_np = np.array(labels)


print("Starting recognizer training...")

face_recognizer.train(images_np, labels_np)
face_recognizer.save("./model/model.XML")

print("Saving trained model...")


