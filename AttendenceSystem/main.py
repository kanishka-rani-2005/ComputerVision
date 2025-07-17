import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime


path=r'AttendenceSystem\ImageAttendence'
images=[]
classNames=[]
myList=os.listdir(path)
# print(myList)

for cl in myList:
    curImg = cv2.imread(os.path.join(path,cl))
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])


print('Images Read Done.')
# print(classNames)


def findEncodings(images):
    encodeList = []
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

# print(images)


def attendence(name):
    path=r'AttendenceSystem\Attendence.csv'
    with open(path,'r+') as f:
        myDataList=f.readlines()
        nameList=[]
        for line in myDataList:
            entry=line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now=datetime.now()      
            date=now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{date}')





encoding=findEncodings(images)
print("Encoding Complete.")

cap=cv2.VideoCapture(0)

while cap.isOpened():
    ret,frame=cap.read()

    if not ret :
        break
    imgS=cv2.resize(frame,(0,0),None,0.25,0.25) # normalize
    imgS=cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

    distance=face_recognition.face_locations(imgS)
    encode=face_recognition.face_encodings(imgS,distance)


    for en,loc in zip(encode,distance):
        match=face_recognition.compare_faces(encoding,en)
        faceDis=face_recognition.face_distance(encoding,en)


        # print(faceDis)
        matchIndex=np.argmin(faceDis)

        if match[matchIndex] :
            name=classNames[matchIndex].upper()
            # print(name)
            y1,x2,y2,x1=loc
            y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,name,(x1+6,y1-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),1)
            attendence(name)


    cv2.imshow('Frame',frame)
    cv2.waitKey(1)  
       

cap.release()
cv2.destroyAllWindows()

    










    