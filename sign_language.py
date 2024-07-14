import cv2
import mediapipe as mp

mpHands = mp.solutions.hands # used to detect the palm and 21 landmark points
mpDrawing = mp.solutions.drawing_utils # used to draws connection between points
hands = mpHands.Hands(min_detection_confidence = 0.8, min_tracking_confidence = 0.5)

tipIDs = [4,8,12,16,20] # pre-assigned values to the tips of the fingers landmarks

video = cv2.VideoCapture(0)

def drawHandLandmarks(frame,handLandmarks):
    if handLandmarks:
        for landmarks in handLandmarks:
            mpDrawing.draw_landmarks(frame,landmarks,mpHands.HAND_CONNECTIONS)

def countFingers(frame,handLandmarks,handNo = 0): # 0 means 1 hand
    if handLandmarks:
        landmarks = handLandmarks[handNo].landmark
        #print(landmarks)
        fingers = []
        for index in tipIDs:
            fingerTipX = landmarks[index].x
            fingerBottomX = landmarks[index - 2].x   
            thumbTipY = landmarks[4].y
            thumbBottomY = landmarks[2].y

            if index != 4:
                if fingerTipX < fingerBottomX:  
                    fingers.append(True)
                    print("finger width id",index,"is closed")
                elif fingerTipX > fingerBottomX:
                    fingers.append(False)
                    print("finger width id", index, "is open")
            
            

        if all(fingers):
                if thumbTipY < thumbBottomY:
                    cv2.putText(frame,"LIKE",(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
                    
                else:
                    cv2.putText(frame,"DISLIKE",(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
        else:
            cv2.putText(frame,"ERROR: HAND SIGNATURE NOT DETECTED, PLEASE USE LEFT HAND",(50,50),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),2)
            

while True:
    ret, frame = video.read()
    frame = cv2.flip(frame,1) # flips horizitonally , 0 flips vertically
    results = hands.process(frame)
    handLandmarks = results.multi_hand_landmarks # get landmark pos. from processd results

    drawHandLandmarks(frame,handLandmarks)
    countFingers(frame,handLandmarks)

    cv2.imshow("Hand Tracker",frame)

    if cv2.waitKey(25) == 32:
        break

video.release()
cv2.destroyAllWindows()