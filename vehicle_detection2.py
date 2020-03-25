






import cv2
import numpy as np
def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

cascade_src = 'cars.xml'
video_src = 'v2.mp4'

cap = cv2.VideoCapture(video_src)

car_cascade = cv2.CascadeClassifier(cascade_src)

while True:
    ret, img = cap.read()
    if (type(img) == type(None)):
        break
    
    img= rescale_frame(img, percent=250)
    kernel = np.ones((3,3),np.float32)/-9
    kernel[2][2]=2

    img = cv2.filter2D(img,-1,kernel)
    


    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)

    for (x,y,w,h) in cars:
    	if w>80 or h>60:
        	cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)      
    
    cv2.imshow('video', img)
    if cv2.waitKey(33) == 27:
        break
    
   

cap.release()
cv2.destroyAllWindows()
