import codecs
import json
import os
from enum import Enum

import cv2

__images__ = None
__labels__ = None
__subjects__ = None

__subject_file__ = 'subjects_all_wEQ.json'
__model_file__ = 'model_all_wEQ.XML'


class ModelConfiguration(Enum):
    """
    Stores different model configurations as enum.
    """
    NoEqualization = 0
    WithEqualization = 1
    AllNoEqualization = 2
    AllWithEqualization = 3
    Standard = 4


class _ModelConfigurationFileSuffix(Enum):
    """
    Stores different file suffixes of model configuration as enum.
    """
    NoEqualization = '_noEQ'
    WithEqualization = '_wEQ'
    AllNoEqualization = '_all_noEQ'
    AllWithEqualization = '_all_wEQ'
    Standard = ''


def set_model_configuration(enum_value):
    """
    Determines which configuration is used to train the model based on the given enum.

    :param enum_value: values used to determine the model configuration (see ModelConfiguration enum)
    """
    global __subject_file__, __model_file__

    if enum_value == ModelConfiguration.Standard:
        __subject_file__ = 'subjects.json'
        __model_file__ = 'model.XML'

    if enum_value == ModelConfiguration.NoEqualization:
        __subject_file__ = 'subjects' + str(_ModelConfigurationFileSuffix.NoEqualization.value) + '.json'
        __model_file__ = 'model' + str(_ModelConfigurationFileSuffix.NoEqualization.value) + '.XML'

    if enum_value == ModelConfiguration.WithEqualization:
        __subject_file__ = 'subjects' + str(_ModelConfigurationFileSuffix.WithEqualization.value) + '.json'
        __model_file__ = 'model' + str(_ModelConfigurationFileSuffix.WithEqualization.value) + '.XML'

    if enum_value == ModelConfiguration.AllNoEqualization:
        __subject_file__ = 'subjects' + str(_ModelConfigurationFileSuffix.AllNoEqualization.value) + '.json'
        __model_file__ = 'model' + str(_ModelConfigurationFileSuffix.AllNoEqualization.value) + '.XML'

    if enum_value == ModelConfiguration.AllWithEqualization:
        __subject_file__ = 'subjects' + str(_ModelConfigurationFileSuffix.AllWithEqualization.value) + '.json'
        __model_file__ = 'model' + str(_ModelConfigurationFileSuffix.AllWithEqualization.value) + '.XML'


def _show_faces(image, faces):
    """
    Takes the face that was found last in the image, draws a rectangle to the given image and shows the result.

    :param image: image that is used to draw the found face on
    :param faces: list of faces that were found in the image
    """
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
        last = i == len(faces) - 1

        if last:
            last = False
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        else:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)

    cv2.namedWindow("Faces found", cv2.WINDOW_NORMAL)

    w, h, d = image.shape
    n_w = int(w / 3)
    n_h = int(h / 3)

    # im_s = cv2.resize(image, (n_w, n_h))

    cv2.resizeWindow('Faces found', n_h, n_w)

    cv2.imshow("Faces found", image)
    cv2.waitKey(0)


def _detect_faces(list_of_images, list_of_labels, list_of_subjects):
    """
    Detects faces in images using haarcascades, crops the section of the image where a face was found and saves
    those information as new lists of images, labels and subjects. These lists are returned.

    :param list_of_images: images that are used to detect faces in
    :param list_of_labels: labels tha
    :param list_of_subjects: subject images belong to as list
    :return: list of images, labels and subjects where faces were found in
    """
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
            scaleFactor=1.1,
            minNeighbors=7,
            minSize=(20, 20)

            # scaleFactor=1.2,
            # minNeighbors=7,
            # minSize=(10, 10))
        )

        #    _show_faces(image, faces)

        if len(faces) == 0:
            continue
        x, y, w, h = faces[-1]

        if list_of_images is not None and list_of_labels is not None:
            list_of_person_images.append(gray[y:y + w, x:x + h])  # Extract detected face
            list_of_person_labels.append(list_of_labels[i])

    return list_of_person_images, list_of_person_labels, list_of_subjects


def _read_training_data(path):
    """

    :param path:
    :return:
    """
    list_of_images = list()
    list_of_labels = list()
    list_of_subjects = list()

    print("Loading training data...")

    # sort files in ascending order to avoid problems in connection with the different file systems of linux and windows
    directory_data = os.listdir(path)
    directory_data.sort()

    for person in directory_data:

        person_directory_data = os.listdir(path + "/" + person)

        for image in person_directory_data:

            pathToImage = path + "/" + person + "/" + image
            img_load = cv2.imread(pathToImage)

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

    save_subjects(subjects, __subject_file__)

    return images, labels, subjects


def get_subjects():
    list_of_subjects = list()

    path = "./training"
    if os.path.exists(path):
        directory_data = os.listdir(path)

        directory_data.sort()

        for person in directory_data:
            if person not in list_of_subjects:
                list_of_subjects.append(person)

        __subjects__ = list_of_subjects
        return list_of_subjects
    else:
        return None


def get_subjects_from_json():
    list_of_subjects = list()
    path = "./model/" + __subject_file__

    with open(path, 'r') as data_file:

        data_dict = json.loads(data_file.read())

        if data_dict is not None:

            return data_dict

        else:

            print("File not found or is empty! " + path)


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

        # scaleFactor=1.2,
        # minNeighbors=7,
        # minSize=(10, 10)
    )

    # show_faces(image, faces)

    if len(faces) == 0:
        return None

    return faces, gray


def extract_face(gray, faces):
    x, y, w, h = faces[-1]

    detected_face = gray[y:y + w, x:x + h]

    return detected_face


def save_subjects(subject_list, filename):
    subject_dic = dict()

    for element in subject_list:
        subject_dic[subject_list.index(element)] = element

    _save_subjects_to_file_(filename, subject_dic)


def _save_subjects_to_file_(filename, dict_of_subjects):
    path = './model/' + filename

    with open(path, 'wb') as outfile:
        json.dump(dict_of_subjects, codecs.getwriter('utf-8')(outfile), indent=4, ensure_ascii=False)
        outfile.flush()
        outfile.close()


def get_model_name():
    return __model_file__
