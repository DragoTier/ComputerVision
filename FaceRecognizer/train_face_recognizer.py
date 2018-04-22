import cv2
import training_data
import numpy as np

# https://www.superdatascience.com/opencv-face-recognition/

print("Starting with opencv version " + cv2.__version__)

# https://docs.opencv.org/3.0-beta/modules/face/doc/facerec/facerec_api.html#createlbphfacerecognizer
face_recognizer = cv2.face_LBPHFaceRecognizer.create()

images, labels, subjects = training_data.get_training_data()

# equalize histogram of all read grayscale images
equalized_images = list()
for image in images:
    equalized_images.add(cv2.equalizeHist(image))  # equalize Histogram


images_np = np.array(images)
equalized_images_np = np.array(equalized_images)
labels_np = np.array(labels)

print("Starting recognizer training...")

# TODO: images_np mit equalized_images_np austauschen
face_recognizer.train(images_np, labels_np)
face_recognizer.save("./model/model.XML")

print("Saving trained model...")


