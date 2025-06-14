import cv2
import pyautogui
import mediapipe as mp
import math

webcam=cv2.VideoCapture(0)

mp_hands=mp.solutions.hands.Hands()
drawing_utils=mp.solutions.drawing_utils

clicking=False
screen_width, screen_height = pyautogui.size()


x1,x2,x3,x4=0,0,0,0
while True:
    _ ,frame=webcam.read()
    frame = cv2.flip(frame, 1)  # Flip the frame horizontally for mirror view

    frame_height,frame_width,_=frame.shape

    rgb_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results=mp_hands.process(rgb_frame)

    hands=results.multi_hand_landmarks
    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(frame,hand)
            landmarks=hand.landmark

            for id,lm in enumerate(landmarks):
                x,y=int(lm.x*frame_width),int(lm.y*frame_height)

                if id==8:
                    cv2.circle(img=frame,center=(x,y),radius=8,color=(255,255,0),thickness=2)
                    x1=x
                    x2=y
                    screen_x=int(lm.x*screen_width-20)
                    screen_y=int(lm.y*screen_height-20)
                if id==4:
                    cv2.circle(img=frame,center=(x,y),radius=8,color=(0,255,0),thickness=2)
                    x3=x
                    x4=y
        pyautogui.moveTo(screen_x,screen_y)
        distance = math.hypot(x1 - x3, x2 - x4)
        if distance < 20:  
            if not clicking:
                pyautogui.click()
                clicking = True
        else:
            clicking = False         


                
    cv2.imshow('Hand Gesture',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
