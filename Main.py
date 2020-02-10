from cv2 import cv2
import numpy
import os
import requests
import time


#additional python files

import Preprocess
import DetectChars
import DetectPlates


##########################
showSteps = True


def main():
    print("Plate Recognition")

    entry = True

    ENDPOINT_ENTRY = 'http://localhost:8080/entry'
    ENDPOINT_EXIT = 'http://localhost:8080/exit'
    blnKNNTrainingSuccessful = DetectChars.loadKNNDataAndTrainKNN()

    if blnKNNTrainingSuccessful == False:  # if KNN training was not successful
        print("\nerror: KNN traning was not successful\n")  # show error message
        return

    #read image file
    imgOriginal = cv2.imread("tab_10.jpg")

    #if could not read image print error message. cv2.imread - do not throw error, if read goea wrong img is None
    if imgOriginal is None:
        print("\nError: Could not read image\n")
        os.system("pause")
        return

    listOfPlates = DetectPlates.detectPlatesInScene(imgOriginal)
    listOfPlates = DetectChars.detectCharsInPlates(listOfPlates)

    if len(listOfPlates) == 0:
        print("No plates found")
    else:
        listOfPlates.sort(key=lambda possiblePlate: len(possiblePlate.strChars), reverse=True)
        licensePlate = listOfPlates[0]
        print("License plate number: " + licensePlate.strChars)

    miliTime = int(round(time.time() * 1000))
    print(miliTime)
    if not showSteps:
        if entry:
            data = {"regNum": licensePlate.strChars,
                    "entryTime": miliTime}
            r = requests.post(url=ENDPOINT_ENTRY, auth=('app', 'app'), json=data)
        else:
            data = {"regNum": licensePlate.strChars,
                    "exitTime": miliTime}
            r = requests.post(url=ENDPOINT_EXIT, auth=('app', 'app'), json=data)
        print(data)
        print(r.content)

    cv2.waitKey(100)
    return


if __name__ == '__main__':
    main()
