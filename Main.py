import cv2
import numpy
import os
import requests

#additional python files

import Preprocess
import DetectChars
import DetectPlates


##########################
showSteps = False


def main():
    print("Plate Recognition")

    ENDPOINT = 'http://localhost:8080/parking/entry'
    blnKNNTrainingSuccessful = DetectChars.loadKNNDataAndTrainKNN()

    if blnKNNTrainingSuccessful == False:  # if KNN training was not successful
        print("\nerror: KNN traning was not successful\n")  # show error message
        return

    #read image file
    imgOriginal = cv2.imread("tab_4.jpg")

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

    data = {"regNum": licensePlate.strChars,
            "entrId": 1568622685000}

    r = requests.post(url=ENDPOINT, auth = ('admin', 'admin'), json=data)

    print(data)
    print(r)



    cv2.waitKey(0)

    return


if __name__ == '__main__':
    main()
