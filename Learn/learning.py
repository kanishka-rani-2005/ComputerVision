import cv2
import matplotlib.pyplot as plt
import numpy as np

# img = cv2.imread('images/image.png')

# if img is None:
#     print("Error: Could not read the image.")
# else:
#     resized_img = cv2.resize(img, (500, 300))

#     cv2.imshow('Resized Image', resized_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# img=cv2.imread('image.png')
# print(img)


# Mulitple Images in one frame
# if img is None:
#     print("Error: Could not read the image.")
# else:
#     resized_img = cv2.resize(img, (200, 200))
#     h=np.hstack((resized_img, resized_img,resized_img))
#     v=np.vstack((h, h))
#     cv2.imshow('Resized Image', v)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()



#slideshow

# import os 

# listname=os.listdir('images')

# for i in listname:
#     img=cv2.imread('images/'+i)
#     resized_img = cv2.resize(img, (500, 300))
#     cv2.imshow('Resized Image', resized_img)
#     if cv2.waitKey(0) :
#         break

# cv2.destroyAllWindows()


#Text on image

# img = cv2.imread('images/image.png')
# img=cv2.resize(img,(300,300))

# text= 'Hello, OpenCV!'
# font=cv2.FONT_HERSHEY_SIMPLEX
# org=(10, 50)  # Starting position of the text
# fontScale=1
# color=(255, 255, 255)  # White color
# thickness=2 
# line_type=cv2.LINE_AA

# img_txt=cv2.putText(img, text, org, font, fontScale, color, thickness, lineType=line_type)

# img_txt=cv2.putText(img_txt, text, org, font, fontScale, color, thickness, lineType=line_type,bottomLeftOrigin=True)


# cv2.imshow('image',img_txt)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#Drawing shapes on image
# old_img = cv2.imread('images/person.png')
# old_img=cv2.resize(old_img,(300,300))


# cv2.imshow('Original Image', old_img)
# # Drawing a circle
# # circle_color = (255, 0, 0)  # Blue color in BGR
# # circle=cv2.circle(old_img, (195, 120), 60, circle_color, -1)  # Filled circle
# # cv2.imshow('Circle Image', circle)

# # Drawing a rectangle
# # rectangle_color = (0, 255, 0)  # Green color in BGR
# # rec=cv2.rectangle(old_img, (140, 20), (250, 180), rectangle_color, 1)
# # cv2.imshow('Rectangle Image', rec)
# # Drawing a line
# # line_color = (0, 0, 255)  # Red color in BGR
# # line=cv2.line(old_img, (140, 70), (230, 70),line_color, 2)  # Line from top-left to bottom-right
# # cv2.imshow('Line Image', line)

# #Draw a elipse
# ellipse_color = (0, 255, 255)
# ellipse=cv2.ellipse(old_img, (190, 100), (80, 40), 90, 0, 360, ellipse_color, -1)  # Filled ellipse
# cv2.imshow('Ellipse Image', ellipse)

# cv2.waitKey(0)

# cv2.destroyAllWindows()


# MERGE images
# img1 = cv2.imread('images/person2.png')
# img2 = cv2.imread('images/person.png')


# img1 = cv2.resize(img1, (300, 300))
# img2 = cv2.resize(img2, (300, 300))

# new_img=cv2.addWeighted(img1, 1, img2, 0.5, 2)  # intensity of first =1 ,intensity of second =1, gamma=0

# new_img=cv2.subtract(img1, img2)  # Subtracting second image from first
# cv2.imshow('Merged Image', new_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# Binary Image # with two colors black and white'

#black->0
#white->1 

# img = cv2.imread('images/dot.png')
# img2= cv2.imread('images/dot2.png')
# img = cv2.resize(img, (300, 300))
# img2 = cv2.resize(img2, (300, 300))

# # new_img=cv2.bitwise_not(img)  # Inverting the colors of the image
# # new_img=cv2.bitwise_and(img, img2)  # AND operation
# new_img=cv2.bitwise_or(img, img)  # OR operation
# # new_img=cv2.bitwise_xor(img, img)  # XOR operation

# cv2.imshow('Binary Image', new_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# Canny Edge Detection
# img= cv2.imread('images/person.png')
# img=cv2.resize(img, (300, 300))

# print(img.shape)  # Print the shape of the image (height, width, channels)

# new_img=cv2.Canny(img, 200, 200,apertureSize=5,L2gradient=True)  # Canny edge detection
# cv2.imshow('Canny Edge Detection', new_img)
# cv2.imshow('Original Image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#Rotation

# img=cv2.imread('images/person.png')
# img=cv2.resize(img, (300, 300))
# w,h=img.shape[0],img.shape[1]

# m=cv2.getRotationMatrix2D((w/2,h/2), 45, 1)  # Center of rotation, angle, scale
# new_img=cv2.warpAffine(img, m, (w, h))  # Apply the rotation matrix to the image

# # new_img=cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)  # Rotate the image 90 degrees clockwise
# # new_img=cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)  # Rotate the image 90 degrees counterclockwise

# cv2.imshow('Rotated Image ', new_img)
# cv2.imshow('Original Image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#Blur
# img=cv2.imread('images/person.png')
# img=cv2.resize(img, (300, 300))

# new_img1=cv2.blur(img, (5, 5))  # Average blur
# new_img2=cv2.GaussianBlur(img, (5, 5), 0)  # Gaussian blur
# new_img3=cv2.medianBlur(img, 5)  # Median blur
# new_img4=cv2.bilateralFilter(img, 9, 75, 75)  # Bilateral filter

# h=np.hstack((new_img1, new_img2, new_img3, new_img4))  # Stack the blurred images horizontally

# cv2.imshow('Blurred Image', h)
# # cv2.imshow('Original Image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# save image
# img=cv2.imread('images/person.png')   
# img1=cv2.imread('images/person2.png')
# img=cv2.resize(img, (300, 300))
# img1=cv2.resize(img1, (300, 300))

# new_img=cv2.addWeighted(img, 1, img1, 0.5, 2)  # intensity of first =1 ,intensity of second =1, gamma=0
# cv2.imwrite('images/merged_image.png', new_img)  # Save the merged image
# cv2.imshow('Merged Image', new_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#play video 

# cap = cv2.VideoCapture('videos/video.mp4')  # Open the video file
# if not cap.isOpened():
#     print("Error: Could not open video.")
# else:
#     while True:
#         ret, frame = cap.read()  # Read a frame from the video
#         if not ret:
#             print("Error: Could not read frame.")
#             break
#         frame = cv2.resize(frame, (600, 400))  # Resize the frame
#         cv2.imshow('Video Frame', frame)  # Display the frame
#         if cv2.waitKey(25) & 0xFF == ord('q'):  # Press 'q' to exit
#             break
#     cap.release()  # Release the video capture object
#     cv2.destroyAllWindows()  # Close all OpenCV windows


# Capture video from webcam
# cap=cv2.VideoCapture(0)
# while True:
#     ret,frame=cap.read()

#     if not ret:
#         print("Error: Could not read frame.")
#         break
#     cv2.imshow('Webcam Frame', frame)  # Display the frame
#     if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
#         break

# cap.release()  # Release the video capture object
# cv2.destroyAllWindows()  # Close all OpenCV windows



# Fast Slow Motion Video

# cap = cv2.VideoCapture('videos/video.mp4')  # Open the video file
# if not cap.isOpened():
#     print("Error: Could not open video.")
# else:
#     while True:
#         ret, frame = cap.read()  # Read a frame from the video
#         if not ret:
#             print("Error: Could not read frame.")

#             break
#         frame = cv2.resize(frame, (600, 400))  # Resize the frame
#         cv2.imshow('Video Frame', frame)  # Display the frame
#         if cv2.waitKey(25) & 0xFF == ord('q'):  # 25 -> 40 fps, 50 -> 20 fps, 100 -> 10 fps  frame per sec

#             break
#     cap.release()  # Release the video capture object
#     cv2.destroyAllWindows()  # Close all OpenCV windows


# Morphological Operations
# Only on Binary Images
# import cv2
# img=cv2.imread('images/dot.png')
# img=cv2.resize(img, (300, 300))
# kernel = np.ones((5, 5), np.uint8)  # Define a kernel for morphological operations
# print(kernel)

# new_img=cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)  # Opening operation
# # new_img=cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)  # Closing operation
# # new_img=cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)  # Gradient operation
# # new_img=cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)  # Top hat operation
# # new_img=cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)  # Black hat operation

# h=np.hstack((img, new_img))  # Stack the original and processed images horizontally
# cv2.imshow('Morphological Operations', h)  # Display the images
# cv2.waitKey(0)
# cv2.destroyAllWindows()



#Image Translation

# img=cv2.imread('images/person.png')
# img=cv2.resize(img,(300,300))


# m=np.float32([[1, 0, 70], [0, 1, 20]])  # Translation matrix (tx, ty)`
# new_img=cv2.warpAffine(img, m, (img.shape[1], img.shape[0]))  # Apply the translation matrix to the image
# cv2.imshow('Translated Image', new_img)  # Display the translated image
# cv2.imshow('Image',img)
# # cv2.imshow('mask',m)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Geometric Transformation


# img = cv2.imread('images/person.png')
# img = cv2.resize(img, (300, 300))
# rows, cols = img.shape[:2]  # Get the dimensions of the image

# # Define the transformation matrix for translation
# tx, ty = 50, 50
# m_translation = np.float32([[1, 0, tx], [0, 1, ty]])  # Translation matrix

# # Apply the translation transformation
# translated_img = cv2.warpAffine(img, m_translation, (cols, rows))
# # Define the transformation matrix for rotation
# angle = 45
# m_rotation = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)  # Rotation matrix

# # Apply the rotation transformation
# rotated_img = cv2.warpAffine(img, m_rotation, (cols, rows))

# cv2.imshow('Original Image', img)
# cv2.imshow('Translated Image', translated_img)
# cv2.imshow('Rotated Image', rotated_img)  
# cv2.waitKey(0)
# cv2.destroyAllWindows()  



# # background subtraction

# img=cv2.imread('images/person.png')
# img = cv2.resize(img, (300, 300))
# bg_subtractor = cv2.createBackgroundSubtractorMOG2()  # Create a background subtractor object
# fg_mask = bg_subtractor.apply(img)  
# cv2.imshow('Foreground Mask', fg_mask)
# cv2.imshow('Original Image', img)  
# cv2.waitKey(0)
# cv2.destroyAllWindows()  


# extract image from video using opencv
# cap=cv2.VideoCapture('videos/video.mp4')
# c=0
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Could not read frame.")
#         break
#     frame = cv2.resize(frame, (600, 400))
#     file_name='videos/org_img_'+str(c)+'.png'
#     cv2.imwrite(file_name, frame)
#     print(f"Saved {file_name}")
#     c += 1
#     cv2.imshow('Video Frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()


#cvtColor method 

# img=cv2.imread('images/person.png')
# img = cv2.resize(img, (300, 300))

# # new_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale
# # new_img=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # Convert the image to HSV color space
# # new_img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert the image to RGB color space
# # new_img=cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # Convert the image to LAB color space
# # new_img=cv2.cvtColor(img, cv2.COLOR_BGR2YUV)  # Convert the image to YUV color space
# # new_img=cv2.cvtColor(img, cv2.COLOR_BGR2XYZ)  # Convert the image to XYZ color space
# # new_img=cv2.cvtColor(img, cv2.COLOR_BGR2LUV)  # Convert the image to LUV color space
# # new_img=cv2.cvtColor(img, cv2.COLOR_BGR2HLS)  # Convert the image to HLS color space
# new_img=cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)  # Convert the image to YCrCb color space
# # new_img=cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)  # Convert the image to BGRA color space

# cv2.imshow('Color Conversion', new_img)  # Display the images
# cv2.imshow('Original Image', img)  # Display the original image
# cv2.waitKey(0)
# cv2.destroyAllWindows()




# #crop image 
# img=cv2.imread('images/person.png')
# img = cv2.resize(img, (300, 300))

# # Define the region of interest (ROI) for cropping
# x, y, w, h = 50, 50, 200, 200  # x, y coordinates and width, height of the ROI
# cropped_img = img[y:y+h, x:x+w]  # Crop the image using slicing

# cv2.imshow('Cropped Image', cropped_img)  # Display the cropped image
# cv2.imshow('Original Image', img)  # Display the original image
# cv2.waitKey(0)
# cv2.destroyAllWindows()
 


# #blank image
# img=cv2.imread('images/person.png')
# img = cv2.resize(img, (300, 300))

# cv2.imshow('Original Image', img)  # Display the original image
# # blank_image = np.ones((300,300,3),np.uint8)*255
# blank_image=np.zeros((300,300,3),np.uint8)*255 
# # Create a blank image with the same shape as the original image
# cv2.imshow('Blank Image', blank_image)  # Display the blank image
# cv2.waitKey(0)
# cv2.destroyAllWindows()  # Close all OpenCV windows



#Get track bar position
# img=np.ones((450, 450, 3), np.uint8)*255
# cv2.namedWindow('Color')
# cv2.createTrackbar('R','Color',0,255,lambda x:x)
# cv2.createTrackbar('G','Color',0,255,lambda x:x)
# cv2.createTrackbar('B','Color',0,255,lambda x:x)

# while True:
#     r=cv2.getTrackbarPos('R','Color')
#     g=cv2.getTrackbarPos('G','Color')
#     b=cv2.getTrackbarPos('B','Color')
#     cv2.imshow('Color',img)
#     if cv2.waitKey(1) & 0xFF == 27: 
#         break # Press 'ESC' to exit
#     img[:]=[b,g,r]

# cv2.destroyAllWindows()




# img=np.ones((450, 450, 3), np.uint8)*255

# cv2.namedWindow('Count')

# cv2.createTrackbar('Count','Count',0,255,lambda x:x)
# while True:
#     count = cv2.getTrackbarPos('Count', 'Count')
#     img[:] = (count, count, count)  # Set the image to a shade of gray based on the trackbar position
#     cv2.imshow('Count', img)
#     if cv2.waitKey(1) & 0xFF == 27: 
#         break  # Press 'ESC' to exit



# cv2.waitKey(0)
# cv2.destroyAllWindows()


#rotate flip traspose 
# import cv2
# import numpy as np

# img=cv2.imread('images/person.png')
# img = cv2.resize(img, (300, 300))

# # img=cv2.flip(img, 0)  # Flip the image vertically
# # img=cv2.flip(img, 1)  # Flip the image horizontally
# # img=cv2.flip(img, -1)  # Flip the image both vertically and horizontally
# # img=cv2.transpose(img)  # Transpose the image (swap rows and columns)
# # img=cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)  # Rotate the image 90 degrees clockwise
# img=cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)  # Rotate the image 90 degrees counterclockwise

# cv2.imshow('Original Image', img)  # Display the original image
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# #savinf video 
# import cv2
# import numpy as np
# # Create a VideoCapture object to read from the webcam
# cap = cv2.VideoCapture(0)
# # Check if the webcam is opened successfully
# if not cap.isOpened():
#     print("Error: Could not open webcam.")
# else:
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for .avi files
#     out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))  # Output file, codec, FPS, frame size

#     while True:
#         ret, frame = cap.read()  # Read a frame from the webcam
#         if not ret:
#             print("Error: Could not read frame.")
#             break
        
#         out.write(frame)  # Write the frame to the output file
#         cv2.imshow('Webcam Frame', frame)  # Display the frame
        
#         if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
#             break

#     cap.release()  # Release the webcam
#     out.release()  # Release the VideoWriter object
#     cv2.destroyAllWindows()  # Close all OpenCV windows



# focus on object without background

# cap=cv2.VideoCapture(0)


# cv2.namedWindow('demo')
# cv2.createTrackbar('lb','demo',0,255,lambda x:x)
# cv2.createTrackbar('lg','demo',0,255,lambda x:x)
# cv2.createTrackbar('lr','demo',0,255,lambda x:x)

# cv2.createTrackbar('ub','demo',0,255,lambda x:x)
# cv2.createTrackbar('ug','demo',0,255,lambda x:x)
# cv2.createTrackbar('ur','demo',0,255,lambda x:x)

# while cap.isOpened():
#     r,frame=cap.read()

#     if r==True:
#         img=cv2.resize(frame,(400,300))
#         hsv_img=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

#         lb=cv2.getTrackbarPos('lb','demo')
#         lg=cv2.getTrackbarPos('lg','demo')
#         lr=cv2.getTrackbarPos('lr','demo')

#         ub=cv2.getTrackbarPos('ub','demo')
#         ug=cv2.getTrackbarPos('ug','demo')
#         ur=cv2.getTrackbarPos('ur','demo')

#         lo=np.array([lb,lg,lr])
#         up=np.array([ub,ug,ur])

#         masks=cv2.inRange(hsv_img,lo,up)  # Create a mask for the specified color range
#         res=cv2.bitwise_and(img,img,mask=masks)  # Apply the mask to the original image

#         cv2.imshow('demo',res)  # Display the result
#         cv2.imshow('mask', masks)  # Display the mask`
#         cv2.imshow('Original', img)  # Display the original image
#         if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
#             break

# cap.release()  # Release the webcam
# cv2.destroyAllWindows()  # Close all OpenCV windows



# Perspective Transformation

# import cv2
# import numpy as np

# paper=cv2.imread('images/paper.png')
# # paper=cv2.resize(paper, (600, 400))

# cv2.circle(paper, (93,429), 5, (0, 0, 255), -1)  # Mark the points on the paper
# cv2.circle(paper, (412,420), 5, (0, 0, 255), -1)
# cv2.circle(paper, (412,643), 5, (0, 0, 255), -1)
# cv2.circle(paper, (93,643), 5, (0, 0, 255), -1)
# w,h=paper.shape[0],paper.shape[1]

# src1=np.float32([[93,429],[830,420],[830,643],[93,643]])
# dest1=np.float32([[0,0],[w,0],[w,h],[0,h]])

# m=cv2.getPerspectiveTransform(src1,dest1)
# new_img=cv2.warpPerspective(paper,m,(400,400))
# cv2.imshow('Scanned',new_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()





# #Threshold
# img=cv2.imread('images/person.png')
# img = cv2.resize(img, (300, 300))

# cv2.imshow('Original Image', img)  # Display the original image`
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale
# # new_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)[1]  # Apply binary thresholding
# new_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)  # Adaptive thresholding
# cv2.imshow('Thresholded Image', new_img)  # Display the thresholded image
# # new_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)[1]  # Inverted binary thresholding
# # new_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)  # Adaptive Gaussian thresholding

# cv2.imshow('Thresholded Image', new_img)
# cv2.waitKey(0)  # Wait for a key press
# cv2.destroyAllWindows()  # Close all OpenCV windows



# img=cv2.imread('images/shape.png')
# img = cv2.resize(img, (400, 400))
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale
# _, thresh = cv2.threshold(gray_img, 200, 255, cv2.THRESH_BINARY)  # Apply binary thresholding

# contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours in the thresholded image

# ar=[]
# for contour in contours:
#     m=cv2.moments(contour)  # Calculate the moments of the contour
#     x=int(m['m10']/m['m00'])
#     y=int(m['m01']/m['m00'])
#     cv2.drawContours(img, contours, -1, (0, 0, 255), 2)  # Draw the contour on the original image
#     cv2.circle(img, (x, y), 5, (0, 255, 0), -1)  # Draw a circle at the centroid
#     a=cv2.contourArea(contour)  # Calculate the area of the contour
#     ar.append(a)  # Append the area to the list

#     ep=0.1*cv2.arcLength(contour, True)  # Calculate the perimeter of the contour
#     approx=cv2.approxPolyDP(contour, ep, True)
#     h=cv2.convexHull(approx)  
#     x,y,w,h=cv2.boundingRect(approx)  # Get the bounding rectangle of the contour
#     cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

# print(ar)  # Print the list of areas
# cv2.imshow('Shape Detection', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()  


# import cv2
# import numpy as np

# cv2.namedWindow('demo')

# cv2.createTrackbar('thr','demo',0,255,lambda x:x)

# cv2.createTrackbar('lb', 'demo', 0, 255, lambda x: x)
# cv2.createTrackbar('lg', 'demo', 0, 255, lambda x: x)
# cv2.createTrackbar('lr', 'demo', 0, 255, lambda x: x)

# cv2.createTrackbar('ub', 'demo', 255, 255, lambda x: x)
# cv2.createTrackbar('ug', 'demo', 255, 255, lambda x: x)
# cv2.createTrackbar('ur', 'demo', 255, 255, lambda x: x)

# cap = cv2.VideoCapture(0)  # Open the webcam

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Could not read frame.")
#         break

#     img = cv2.resize(frame, (400, 300))
#     img=cv2.flip(img,1)  # Flip the image horizontally
#     hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     thr = cv2.getTrackbarPos('thr', 'demo')

#     lb = cv2.getTrackbarPos('lb', 'demo')
#     lg = cv2.getTrackbarPos('lg', 'demo')
#     lr = cv2.getTrackbarPos('lr', 'demo')

#     ub = cv2.getTrackbarPos('ub', 'demo')
#     ug = cv2.getTrackbarPos('ug', 'demo')
#     ur = cv2.getTrackbarPos('ur', 'demo')

#     lo = np.array([lb, lg, lr])
#     up = np.array([ub, ug, ur])

#     masks = cv2.inRange(hsv_img, lo, up)  # Create a mask for the specified color range
#     res = cv2.bitwise_and(img, img, mask=masks)  # Apply the mask to the original image
#     # fr=cv2.bitwise_not(res)  # Invert the mask to get the background

#     thi=cv2.threshold(masks, thr, 255, cv2.THRESH_BINARY)[1] # Apply binary thresholding to the mask

#     cnt,hr=cv2.findContours(thi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

#     cv2.drawContours(img, cnt, -1, (0, 0, 255), 2)  # Draw the contours on the result image

#     cv2.imshow('thresh', thi)  # Display the result image
#     cv2.imshow('demo', res)  
#     cv2.imshow('mask', masks)  
#     cv2.imshow('Original', img)  

#     if cv2.waitKey(1) & 0xFF == ord('q'):  
#         break

# cap.release()  
# cv2.destroyAllWindows()  


# Corner in an image
# import cv2
# import numpy as np

# img=cv2.imread('images/person.png')
# img = cv2.resize(img, (300, 300))


# cv2.imshow('Original Image', img)  # Display the original image
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale      
# gray_img = np.float32(gray_img)  # Convert the grayscale image to float32 type
# new_img=cv2.cornerHarris(gray_img, 2, 3, 0.04)  # Apply Harris corner detection
# new_img=cv2.dilate(new_img, None)  # Dilate the corners to make them more visible

# img[new_img > 0.01 * new_img.max()] = [0, 0, 255]  # Mark the corners in red on the original image
# cv2.imshow('Harris Corners', img)  # Display the image with corners marked

# cv2.waitKey(0)  # Wait for a key press
# cv2.destroyAllWindows()  # Close all OpenCV windows




# import numpy as np

# img=cv2.imread('images/person.png')
# img = cv2.resize(img, (300, 300))


# cv2.imshow('Original Image', img)  # Display the original image
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale      

# corners = cv2.goodFeaturesToTrack(gray_img, maxCorners=100, qualityLevel=0.01, minDistance=10)  
# corners = np.int64(corners)  # Convert the corners to integer type

# for i in corners:
#     x, y = i.ravel()  # Flatten the corner coordinates
#     cv2.circle(img, (x, y), 2, (0, 255, 0), -1)  # Draw a circle at each corner

# cv2.imshow('Good Features to Track', img)  # Display the image with corners marked

# cv2.waitKey(0)  # Wait for a key press
# cv2.destroyAllWindows()  # Close all OpenCV windows

# import cv2
# import numpy as np

# img=cv2.imread('images/person.png')
# # img = cv2.resize(img, (300, 300))
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert the imag

# # Load the Haar Cascade classifier for face detection
# face_cascade = cv2.CascadeClassifier('learn/haarcascade_frontalface_default.xml')
# # Detect faces in the image
# faces = face_cascade.detectMultiScale(gray_img, 1.1, 5)
# # Draw rectangles around the detected faces
# for (x, y, w, h) in faces:
#     cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw a rectangle around each face
#     # Display the image with faces marked
# cv2.imshow('Face Detection', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()  


# import cv2
# import numpy as np

# def click_event(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:  # Check if the left mouse button is clicked
#         print(f"Left click at ({x}, {y})")  # Print the coordinates of the clicked point
#         cv2.circle(img, (x, y), 5, (0, 255, 0), -1)  # Draw a circle at the clicked point
#         cv2.putText(img, f"({x}, {y})", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
#         cv2.imshow('demo', img)  # Update the displayed image

#     elif event == cv2.EVENT_RBUTTONDOWN:
#         print(f"Right click at ({x}, {y})")
#         b=img[y, x, 0]
#         g=img[y, x, 1]
#         r=img[y, x, 2]
#         cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
#         cv2.putText(img, f"({b}, {g}, {r})", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
#         cv2.imshow('demo', img)  # Update the displayed image


# img=cv2.imread('images/person.png')
# cv2.imshow('demo', img)  # Display the image
# cv2.setMouseCallback('demo',click_event)

# cv2.waitKey(0)
# cv2.destroyAllWindows()  


#Reverse a video
# import cv2
# import numpy as np
# cap = cv2.VideoCapture('videos/video.mp4')  # Open the video file
# c=1
# l=[]
# while cap.isOpened():
#     ret, frame = cap.read()  # Read a frame from the video
#     if not ret:
#         print("Error: Could not read frame.")
#         break
#     filename="videos/frame"+c+".png"
#     c=c+1
#     l.append(filename)
#     cv2.imwrite(filename, frame)  # Save the frame as an image file
#     cv2.imshow('Original Frame', frame)  # Display the original frame
#     if cv2.waitKey(25) & 0xFF == ord('q'):  # Press 'q' to exit
#         break
# l.reverse()
# for i in l:
#     img=cv2.imread(i)
#     cv2.imshow('Reversed Frame', img)  # Display the reversed frame
#     if cv2.waitKey(25) & 0xFF == ord('q'):  # Press 'q' to exit
#         break
# cap.release()  # Release the video capture object
# cv2.destroyAllWindows()  # Close all OpenCV windows


#eye Detection 

# img=cv2.imread('images/person.png')
# img = cv2.resize(img, (300, 300))
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# eye_cascade = cv2.CascadeClassifier('learn/haarcascade_eye.xml')  # Load the Haar Cascade classifier for eye detection
# eyes = eye_cascade.detectMultiScale(gray, 1.1, 2)
# for (x, y, w, h) in eyes:
#     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)  # Draw a rectangle around each detected eye 


# cv2.imshow('Demo', img)  # Display the original image
# cv2.waitKey(0) 
# cv2.destroyAllWindows()


#smile detection

# img=cv2.imread('images/person.png')
# gry=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# sm=cv2.CascadeClassifier(r'learn\haarcascade_smile.xml')
# f=sm.detectMultiScale(gry,1.8,10)


# for (x, y, w, h) in f:
#     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)  # Draw a rectangle around each detected eye 


# cv2.imshow('Demo', img)  # Display the original image
# cv2.waitKey(0) 
# cv2.destroyAllWindows()

# import mediapipe as mp

# mp_face_detection = mp.solutions.face_detection
# mp_drawing = mp.solutions.drawing_utils

# face_det=mp_face_detection.FaceDetection(min_detection_confidence=0.2,model_selection=0)

# cap = cv2.VideoCapture(0)
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Could not read frame.")
#         break
#     frame = cv2.flip(frame, 1)  # Flip the frame horizontally
#     results = face_det.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Process the frame for face detection
#     if results.detections:
#         for detection in results.detections:
#             mp_drawing.draw_detection(frame,detection)  # Draw the detected face
#             x, y, w, h = int(detection.location_data.relative_bounding_box.xmin * frame.shape[1]),int(detection.location_data.relative_bounding_box.ymin * frame.shape[0]), \
#                           int(detection.location_data.relative_bounding_box.width * frame.shape[1]),int(detection.location_data.relative_bounding_box.height * frame.shape[0])
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw rectangle around detected face
#             cv2.putText(frame, f'{int(detection.score[0] * 100)}%', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Display confidence score
#     cv2.imshow('Face Detection', frame) 
#     if cv2.waitKey(1) & 0xFF == ord('q'): 
#         break

# cap.release()  
# cv2.destroyAllWindows()   