import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math


final_output = ""
letters = []
count_frames = 20
lett = ""


cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("model/fall_keras_model.h5", "model/fall_labels.txt")

offset = 50
imgSize = 300
counter = 0

count_frames = 20
final_output = ""
letters = []


labels = ["A", "B", "back", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
          "N", "O", "P", "Q", "R", "S", "space", "T", "U", "V", "W", "X", "Y", "Z"]           #back, space, j, z


while True:
    success, img = cap.read()
    hands = detector.findHands(img, draw=False)


    filtered = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filtered = cv2.GaussianBlur(filtered, (5, 5), 2)
    filtered = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    ret, filtered = cv2.threshold(filtered, 170, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    #cv2.imshow("Original", img)


    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize), np.uint8)*255
        imgCrop = filtered[y-offset : y+h+offset, x-offset : x+w+offset]
        imgCropShape = imgCrop.shape

        aspectRatio = h/w

        try:
            if aspectRatio > 1:
                k = imgSize/h
                wCal = math.ceil(k*w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape

                wGap = math.ceil((imgSize-wCal)/2)
                imgWhite[:, wGap:wCal+wGap] = imgResize
                gray2rgb = cv2.cvtColor(imgWhite, cv2.COLOR_GRAY2RGB)

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape

                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
                gray2rgb = cv2.cvtColor(imgWhite, cv2.COLOR_GRAY2RGB)

            prediction, index = classifier.getPrediction(gray2rgb)
            print(labels[index])


            count_frames -= 1
            lett = lett.replace(lett, "")

            if count_frames == 0:
                #print(str(final_output))
                count_frames = 20
                lett = max(letters, key=letters.count)
                letters.clear()

                if lett == "space":
                    final_output += " "
                elif lett == "back":
                    # final_output.pop(-1)
                    final_output = final_output[:-1]
                else:
                    final_output += lett
            else:
                letters.append(labels[index])

            #print(prediction)
            #print(letters)
            cv2.putText(
                img,
                final_output,
                (10,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (255, 0, 255),
                3)
            #print(final_output)


            if (x-offset > 0 and x+offset < img.shape[1]  and  y-offset > 0  and  y+offset < img.shape[0]):
                #cv2.imshow("Filtered", filtered)
                #cv2.imshow("Cropped", imgCrop)
                #print(str(final_output))
                cv2.imshow("Final", imgWhite)

        except:
            print("ERROR: Hand out of frame")

    cv2.imshow("Original", img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()


