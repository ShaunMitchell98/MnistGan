import os


def emptyDirectory(directoryName):

    for fileName in os.listdir(directoryName):
        os.remove(os.path.join(directoryName, fileName))
