import cv2
from cvzone.HandTrackingModule import HandDetector
import time

class Button:
    def __init__(self,pos,width,height,value):
        self.pos = pos
        self.width = width
        self.height = height
        self.value=value

    def draw(self, img):
        cv2.rectangle(img, self.pos, (self.pos[0]+self.width, self.pos[1]+self.height), (255, 255, 255), cv2.FILLED)  # Draw a rectangle at the top
        cv2.rectangle(img, self.pos, (self.pos[0]+self.width, self.pos[1]+self.height ), (50, 50, 50), 2)  # Draw a border around the rectangle

        cv2.putText(img,self.value,(self.pos[0]+40,self.pos[1]+60),cv2.FONT_HERSHEY_PLAIN, 2, (50,50,50), 2)

    def checkClick(self, x, y):
        if self.pos[0] < x < self.pos[0] + self.width and self.pos[1] < y < self.pos[1] + self.height:
            cv2.rectangle(img, self.pos, (self.pos[0]+self.width, self.pos[1]+self.height), (255, 255, 255), cv2.FILLED)  # Draw a rectangle at the top
            cv2.rectangle(img, self.pos, (self.pos[0]+self.width, self.pos[1]+self.height ), (50, 50, 50), 2)  # Draw a border around the rectangle

            cv2.putText(img,self.value,(self.pos[0]+40,self.pos[1]+60),cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0), 5)
            return True
        return False

        


cap= cv2.VideoCapture(0)
cap.set(3, 1080)  # Set width
cap.set(4, 720)   # Set height
detector = HandDetector(detectionCon=0.8, maxHands=1)


# Creating buttons
buttonValue=[['7', '8', '9', '/'],
              ['4', '5', '6', '*'],
              ['1', '2', '3', '-'],
              ['0', '.', '=', '+']]
buttonList=[]
for x in range(4):
    for y in range(4):
        xpos=x*100 + 700
        ypos=y*100 + 150
        buttonList.append(Button((xpos, ypos), 100, 100, buttonValue[y][x]))

#Variables
myEqn=''
delayCounter = 0



# Initialize the video capture and hand detector
while True :
    success, img = cap.read()   
    if not success:

        break
    img=cv2.flip(img,1)
    hands, img = detector.findHands(img,flipType=False)  # Find the hands in the image

    #Draw all buttons
    cv2.rectangle(img, (700, 50), (700+400, 50+100), (255, 255, 255), cv2.FILLED)  # Draw a rectangle at the top
    cv2.rectangle(img, (700, 50), (700+400, 50+100), (50, 50, 50), 2)  # Draw a border

    for button in buttonList:
        button.draw(img)

    button=Button((700, 550), 400, 100, 'Clear')
    button.draw(img)
    #Check for hand
    if hands:
        lmList = hands[0]['lmList']
        if len(lmList) >= 13:
            length,_, img = detector.findDistance(lmList[8][:2], lmList[12][:2], img)

            # Detect click (fingers close together)
            x,y=lmList[8][0], lmList[8][1]
            if length < 60 :
                if button.checkClick(x, y) and delayCounter == 0 :
                    val= button.value
                    if val == 'Clear':
                        myEqn = ''
                        delayCounter = 1  # Set the cooldown counter to prevent multiple clicks
                for i,button in enumerate(buttonList):
                    
                    if button.checkClick(x, y) and delayCounter == 0:
                        val=buttonValue[int(i%4)][int(i/4)]
                        if val == '=':
                            try:
                                myEqn = str(eval(myEqn))
                            except Exception as e:
                                raise e
                        
                        else:
                            myEqn += val
                        
                        delayCounter = 1
                          # Reset the cooldown counter
    # Avoid duplicate
    if delayCounter!=0:
        delayCounter += 1
        if delayCounter > 10:
            delayCounter = 0                 
                        
          
    #Display the Result
    cv2.putText(img,myEqn,(710,110),cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0), 2)


    # Display the image with detected hands
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()