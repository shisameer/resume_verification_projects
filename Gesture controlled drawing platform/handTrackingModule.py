import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode=False, maxHands = 2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionCon,
                                        self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        #print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handlns in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handlns, self.mpHands.HAND_CONNECTIONS)
        return img
    
    def findPosition(self, img, handNo=0, draw= True ):
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
        
            for id, ln in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int((ln.x)*w), int((ln.y)*h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            
        return self.lmList
    
    def fingerUp(self):
        fingup = []
        tip = [8, 12, 16, 20]
        if(self.lmList[17][1] > self.lmList[4][1]):
            if(self.lmList[4][1] < self.lmList[3][1]):
                fingup.append(1)
            else:
                fingup.append(0)
        else:
            if(self.lmList[4][1] < self.lmList[3][1]):
                fingup.append(0)
            else:
                fingup.append(1)
        
        for i in tip:
            if(self.lmList[i][2] < self.lmList[i-2][2]):
                fingup.append(1)
            else:
                fingup.append(0)
        
        return fingup
        
    

def main():
    pTime= 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        
        cv2.putText(img, str(int(fps)), (18, 78), cv2.FONT_HERSHEY_PLAIN, 3, 
                    (150, 0, 255), 3)
        
        cv2.imshow("image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()