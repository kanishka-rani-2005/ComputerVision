import cv2
import winsound

webcam = cv2.VideoCapture(0)

_, img1 = webcam.read()
img1 = cv2.flip(img1, 1)

while True:
    _, img2 = webcam.read()
    img2 = cv2.flip(img2, 1)

    diff = cv2.absdiff(img1, img2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        if cv2.contourArea(c) < 1000:
            continue
        print('hello')
        winsound.Beep(1000, 300)

    cv2.imshow('Security Camera', thresh)

    img1 = img2

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
