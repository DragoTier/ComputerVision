import cv2
import training_data
import numpy as np

# https://www.superdatascience.com/opencv-face-recognition/

print("Starting with opencv version " + cv2.__version__)

# https://docs.opencv.org/3.0-beta/modules/face/doc/facerec/facerec_api.html#createlbphfacerecognizer
face_recognizer = cv2.face_LBPHFaceRecognizer.create()

# SET MODEL CONFIG
training_data.set_model_configuration(training_data.ModelConfiguration.AllWithEqualization)

images, labels, subjects = training_data.get_training_data()

# equalize histogram of all read grayscale images
equalized_images = list()
for image in images:
    equalized_images.append(cv2.equalizeHist(image))  # equalize Histogram

images_np = np.array(images)
equalized_images_np = np.array(equalized_images)
labels_np = np.array(labels)

print("Starting recognizer training...")

if images_np is not None and equalized_images_np is not None and labels_np is not None:

    face_recognizer.train(equalized_images_np, labels_np)
    face_recognizer.save("./model/" + training_data.get_model_name())

    print("Saving trained model...")

else:

    print("No training data!")
