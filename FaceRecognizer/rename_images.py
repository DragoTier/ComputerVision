import os


def __rename_training_data__(path):
    temp_directory_data = os.listdir(path)
    counter = 0

    for image in temp_directory_data:
        os.rename(path + "/" + image, path + "/" + "_TEMP_" + str(counter) + "." + image.split(".")[1])
        counter += 1

    counter = 0
    directory_data = os.listdir(path)
    print(directory_data)

    for image in directory_data:
        os.rename(path + "/" + image, path + "/" + str(counter) + "." + image.split(".")[1])
        counter += 1


__rename_training_data__("./training/merkel")