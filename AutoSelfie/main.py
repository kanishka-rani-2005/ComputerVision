import cv2
import mediapipe as mp
import winsound


face_mesh=mp.solutions.face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

x1,x2,y1,y2=0,0,0,0
cap=cv2.VideoCapture(0)

while cap.isOpened():
    r,frame=cap.read()
    if not r:
        break

    frame=cv2.flip(frame,1)
    fh,fw=frame.shape[:2]
    rgb_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    output=face_mesh.process(rgb_frame)
    
    landmark_points= output.multi_face_landmarks
    if landmark_points:
        landmark_points = landmark_points[0].landmark
        for id ,landmark in enumerate(landmark_points):
            x, y = int(landmark.x * fw), int(landmark.y * fh)

            if id==43: 
                x1,y1=x,y
            elif id==287:
                x2,y2=x,y

        dist=int(((x2-x1)**2+(y2-y1)**2)**0.5)
        # print(dist)
        if dist>90:
            winsound.Beep(2500, 1000)
            cv2.imwrite('AutoSelfie/selfie.png',frame)
            cv2.waitKey(100)          
                
    cv2.imshow('Frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

