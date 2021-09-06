import cv2
import mediapipe as mp
import time
import handTrackingModule as htm
import os
import numpy as np
wCam, hCam = 1920, 1080
# 1280, 720
# 1920, 1080
pTime= 0
cTime = 0
url = 'http://192.168.43.1:8080/video'
cap = cv2.VideoCapture(url)
imgCanvas = np.zeros((hCam, wCam, 3), np.uint8)
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.handDetector(detectionCon=0.8, trackCon=0.8)
px, py = 0, 0
tx, ty = 0, 0
color = (255, 0, 255)
while True:
    success, img = cap.read()
    #img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    
    if len(lmList) != 0:
        fingup = detector.fingerUp()
        #print(fingup)
        cx, cy = lmList[8][1:]
        tx, ty = lmList[4][1:]
        if(px == 0 and py == 0):
            px, py = cx, cy
        if(tx == 0 and ty == 0):
            tx, ty = lmList[4][1:]
        if(fingup[4] == 1):
            color = (180, 60, 0)
        elif(fingup[3] == 1):
            color = (0, 223, 255)
        elif(fingup[2] == 1):
            color = (255, 0, 255)
        if fingup[1] and (not fingup[0] and fingup.count(1) == 1):
            cv2.circle(img,(cx, cy), 9, color, cv2.FILLED)
            cv2.line(imgCanvas, (px, py), (cx, cy), color, 3)
        if fingup[0] and fingup.count(1) == 1:
            cv2.circle(img,(lmList[4][1:]), 40, (255, 255, 255), cv2.FILLED)
            cv2.line(imgCanvas, (px, py), lmList[4][1:], (0, 0, 0), 80)
        px, py = cx, cy
        tx, ty = lmList[4][1:]
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    
    cv2.putText(img, str(int(fps)), (18, 78), cv2.FONT_HERSHEY_PLAIN, 3, 
                (150, 0, 255), 3)
    
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img,imgInv)
    img = cv2.bitwise_or(img,imgCanvas)
    #img = cv2.addWeighted(img,1,imgCanvas,1,0)
    cv2.imshow("image", img)
    #cv2.imshow("canvas", imgCanvas)
    cv2.waitKey(1)