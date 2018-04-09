# from https://github.com/informramiz/opencv-face-recognition-python/blob/master/OpenCV-Face-Recognition-Python.py

import cv2 as cv
import os
import numpy as np

# constants directory strings
TRAINING_DIRECTORY_PATH = "training-data"
TEST_DIRECTORY_PATH = "test-data"
# better results?
CASCADE_CLASSIFIER_DIRECTORY_PATH = "cascades-data/haarcascades_cuda/haarcascade_frontalface_default.xml"


def detect_face(img):
    """

    :param img:
    :return:
    """
    # convert the test image to gray scale as opencv face detector expects gray images
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    face_cascade = cv.CascadeClassifier(CASCADE_CLASSIFIER_DIRECTORY_PATH)

    # let's detect multiscale images(some images may be closer to camera than others)
    # result is a list of faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=8)

    # if no faces are detected then return original img
    if len(faces) == 0:
        return None, None

    # under the assumption that there will be only one face,
    # extract the face area
    x, y, w, h = faces[0]

    # return only the face part of the image
    return gray[y:y + w, x:x + h], faces[0]


# this function will read all persons' training images, detect face from each image
# and will return two lists of exactly same size, one list
# of faces and another list of labels for each face
def build_directory_path(directory_path, dir_name):
    return directory_path + "/" + dir_name


def prepare_training_data(data_folder_path):
    # get the directories (one directory for each subject) in data folder
    dirs = os.listdir(data_folder_path)

    faces = []
    labels = []
    subjects = []

    count_folders = 0

    # let's go through each directory and read images within it
    for dir_name in dirs:

        label = count_folders

        # build path of directory containing images for current subject subject
        subject_dir_path = build_directory_path(data_folder_path, dir_name)

        # get the images names that are inside the given subject directory
        subject_images_names = os.listdir(subject_dir_path)

        subjects.append(dir_name)

        count_folders += 1

        for image_name in subject_images_names:

            # ignore system files
            if image_name.startswith("."):
                continue

            image_path = build_directory_path(subject_dir_path, image_name)

            image = cv.imread(image_path)

            face, rect = detect_face(image)

            if face is not None:
                faces.append(face)
                labels.append(label)
            else:
                print(f"Something went wrong! {image_name}")

    return faces, labels, subjects


# according to given (x, y) coordinates and
# given width and height
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


# function to draw text on give image starting from
# passed (x, y) coordinates.
# def draw_text(img, text, x, y):
#     cv.putText(img, text, (x, y), cv.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


def predict(test_image):
    # make a copy of the image as we don't want to change original image
    img = test_image.copy()
    # detect face from the image
    face, rect = detect_face(img)

    if face is None or rect is None:
        print("No face recognized!")
        print("-----------------------")
        return None
    else:
        # predict the image using our face recognizer
        label, probability = face_recognizer.predict(face)
        print(f"With probability of {probability}%")
        # get name of respective label returned by face recognizer
        label_text = subjects[label]

        print(f"It is: {label_text}")
        print("-----------------------")

        # draw a rectangle around face detected
        draw_rectangle(img, rect)
        # draw name of predicted person
        # draw_text(img, label_text, rect[0], rect[1] - 5)

        return img


def get_files_count(data_folder_path):
    dirs = os.listdir(data_folder_path)
    return len(dirs)


def predict_test_images():
    for i in range(1, get_files_count(TEST_DIRECTORY_PATH) + 1):
        test_img = cv.imread(f"{TEST_DIRECTORY_PATH}/{i}.jpg")
        predicted_img = predict(test_img)

        if predicted_img is not None:
            cv.imshow("TEst", cv.resize(predicted_img, (400, 500)))
            cv.waitKey(0)
            cv.destroyAllWindows()


print("Preparing data...")
faces_test, labels, subjects = prepare_training_data(TRAINING_DIRECTORY_PATH)
print("Data prepared")

# print total faces and labels
print("Total faces: ", len(faces_test))
print("Total labels: ", len(labels))

face_recognizer = cv.face_LBPHFaceRecognizer.create()

face_recognizer.train(faces_test, np.array(labels))

print("Predicting images...")


predict_test_images()

print("Prediction complete")
