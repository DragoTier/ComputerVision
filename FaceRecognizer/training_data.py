import cv2
import os
import numpy as np

__images__ = None
__labels__ = None
__subjects__ = None


def _show_faces(image, faces):
    first = True
    last = False

    # for (x, y, w, h) in faces:
    #     if first:
    #         first = False
    #         cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    #     else:
    #         cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)

    for i in range(0, len(faces)):
        (x, y, w, h) = faces[i]
        last = i == len(faces)-1

        if last:
            last = False
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        else:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)

    cv2.namedWindow("Faces found", cv2.WINDOW_NORMAL)

    w, h, d = image.shape
    n_w = int(w / 3)
    n_h = int(h / 3)

    im_s = cv2.resize(image, (n_h, n_w))

    cv2.imshow("Faces found", im_s)
    cv2.waitKey(0)


def _detect_faces(list_of_images, list_of_labels, list_of_subjects):

    list_of_person_images = list()
    list_of_person_labels = list()

    print("Preparing training data...")

    for i in range(0, len(list_of_images)):

        image = list_of_images[i]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier('./data/haarcascades_cuda/haarcascade_frontalface_default.xml')
        # lbpcascades/lbpcascade_frontalface.xml
        # haarcascades/haarcascade_frontalface_default.xml

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=7,
            minSize=(10, 10)
            # maxSize=(70, 70)
        )

    #    _show_faces(image, faces)

        if len(faces) == 0:
            continue
        x, y, w, h = faces[-1]

        if list_of_images is not None and list_of_labels is not None:
            list_of_person_images.append(gray[y:y+w, x:x+h])     # Extract detected face
            list_of_person_labels.append(list_of_labels[i])

    return list_of_person_images, list_of_person_labels, list_of_subjects


def _read_training_data(path):
    list_of_images = list()
    list_of_labels = list()
    list_of_subjects = list()

    print("Loading training data...")

    directory_data = os.listdir(path)
    directory_data.sort()

    for person in directory_data:

        person_directory_data = os.listdir(path + "/" + person)

        for image in person_directory_data:

            pathToImage = path + "/" + person + "/" + image
            img_load = cv2.imread(pathToImage)

            print(str(directory_data))

            if list_of_images is not None and list_of_labels is not None:
                list_of_images.append(img_load)
                if person not in list_of_subjects:
                    list_of_subjects.append(person)
                list_of_labels.append(list_of_subjects.index(person))

    # for image in directory_data:
    #
    #     image_load = cv2.imread(path + "/" + image)
    #     list_of_images = list_of_images.append(image_load)
    #
    #     list_of_labels = list_of_labels.append(path.split("/")[-1])
    #

    images, labels, subjects = _detect_faces(list_of_images, list_of_labels, list_of_subjects)

    return images, labels, subjects


def get_training_data():
    images, labels, subjects = _read_training_data("./training")

    print("Images: " + str(len(images)))
    print("Labels: " + str(len(labels)))
    print("Subjects: " + str(len(subjects)))

    global __images__, __labels__, __subjects__
    __images__, __labels__, __subjects__ = images, labels, subjects

    return images, labels, subjects


def get_subjects():
    list_of_subjects = list()
    directory_data = os.listdir("./training")
    directory_data.sort()

    print(directory_data)

    for person in directory_data:
        if person not in list_of_subjects:
            list_of_subjects.append(person)

    __subjects__ = list_of_subjects
    return list_of_subjects


def detect_face(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier('./data/haarcascades_cuda/haarcascade_frontalface_default.xml')
    # lbpcascades/lbpcascade_frontalface.xml
    # haarcascades/haarcascade_frontalface_default.xml

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=7,
        minSize=(20, 20)
        # maxSize=(70, 70)
    )

    # show_faces(image, faces)

    if len(faces) == 0:
        return None

    return faces, gray

def extract_face(gray, faces):

    x, y, w, h = faces[-1]

    detected_face = gray[y:y + w, x:x + h]

    return detected_face