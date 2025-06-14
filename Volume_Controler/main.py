import cv2
import mediapipe as mp
import pyautogui


x1,y1,x2,y2=0,0,0,0

webcam=cv2.VideoCapture(0)
mp_hands = mp.solutions.hands.Hands()
drawing_utils=mp.solutions.drawing_utils


while True:
    ret ,frame =webcam.read()
    frame=cv2.flip(frame,1)

    frame_height,frame_width,_=frame.shape


    rgb_img=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results = mp_hands.process(rgb_img)


    hands=results.multi_hand_landmarks
    if hands:
        for hand in hands:
            #draw landmarks around hands
            drawing_utils.draw_landmarks(frame, hand)
            landmarks=hand.landmark

            #get the tip of the index finger
            for id,lm in enumerate(landmarks):
                x = int(lm.x * frame_width)
                y = int(lm.y * frame_height)
                # fore finger
                if id==8:
                    # draw circle around fore finger
                    cv2.circle(img=frame,center=(x,y),radius=8,color=(0,255,255),thickness=2)
                    x1,y1=x,y
                # thumb 
                if id==4:
                    # draw circle around thumb
                    cv2.circle(img=frame,center=(x,y),radius=8,color=(0,255,0),thickness=2)
                    x2,y2=x,y

        # calcualting distance between thumb and fore finger
        dist=((x2-x1)**2 +(y2-y1)**2)**0.5//4

        if dist>20:
            # print("Distance between fingers is greater than 30")
            pyautogui.press('volumeup')
        else:
            # print("Distance between fingers is less than 30")
            pyautogui.press('volumedown')

        cv2.line(frame,(x1,y1),(x2,y2),(255,255,0),2)
            
            
    cv2.imshow('Hand gesture ',frame)

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()