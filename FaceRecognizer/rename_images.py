import os


def __rename_training_data__(path):
    """
    Renames all files at a given path to have numerical names only
    :param path: path where files will be renamed
    """
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


# Insert directory to rename here
# __rename_training_data__("./training/merkel")
