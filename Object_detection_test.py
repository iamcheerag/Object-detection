import numpy as np
import cv2
import pyttsx3

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
blind_stick_cascade = cv2.CascadeClassifier('blind_stick_cascade.xml')
#mobile_cascade = cv2.CascadeClassifier('mobile_cascade.xml')
mobile_charger_cascade = cv2.CascadeClassifier('mobile_charger_cascade.xml')
#plate_cascade = cv2.CascadeClassifier('plate_cascade.xml')
radio_cascade = cv2.CascadeClassifier('radio_cascade.xml')
spoon_cascade = cv2.CascadeClassifier('spoon_cascade.xml')
toothbrush_cascade = cv2.CascadeClassifier('toothbrush_cascade.xml')
#vehicle_cascade = cv2.CascadeClassifier('vehicle_cascade.xml')


engine = pyttsx3.init()

cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray,1.3, 5)
    word1="Human face detected"
    word2="Blind Stick Detected"
    word3="Mobile Phone detected"
    word4="Mobile Charger detected"
    word5="Plate Detected"
    word6="Radio Detected"
    word7="Spoon Detected"
    word8="Toothbrush Detected"
    word9="Vehicle Detected"
    
    for (x,y,w,h) in face:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        font=cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,word1,(x-w,y+h),font,1,(255,255,0),2,cv2.LINE_AA)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        engine.say(word1)
        engine.setProperty('rate',100)
        engine.runAndWait()
        
    stick = blind_stick_cascade.detectMultiScale(gray,1.3, 5) 
    for (ex,ey,ew,eh) in stick:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        font=cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,word2,(ex-ew,ey+eh),font,1,(255,255,0),2,cv2.LINE_AA)
        engine.say(word2)
        engine.setProperty('rate',100)
        engine.runAndWait()
        
    mobile_charger = mobile_charger_cascade.detectMultiScale(gray,1.3, 5)
    for (mc_x,mc_y,mc_w,mc_h) in mobile_charger:
        cv2.rectangle(roi_color,(mc_x,mc_y),(mc_x+mc_w,mc_y+mc_h),(0,255,0),2)
        font=cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,word4,(mc_x-mc_w,mc_y+mc_h),font,1,(255,255,0),2,cv2.LINE_AA)
        engine.say(word4)
        engine.setProperty('rate',100)
        engine.runAndWait()
        

    radio = radio_cascade.detectMultiScale(gray,1.3, 5)
    for (rx,ry,rw,rh) in radio:
        cv2.rectangle(roi_color,(rx,ry),(rx+rw,ry+rh),(0,255,0),2)
        font=cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,word6,(rx-rw,ry+rh),font,1,(255,255,0),2,cv2.LINE_AA)
        engine.say(word6)
        engine.setProperty('rate',100)
        engine.runAndWait()
        
    spoon = spoon_cascade.detectMultiScale(gray,1.3, 5)
    for (sx,sy,sw,sh) in spoon:
        cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(0,255,0),2)
        font=cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,word7,(sx-sw,sy+sh),font,1,(255,255,0),2,cv2.LINE_AA)
        engine.say(word7)
        engine.setProperty('rate',100)
        engine.runAndWait()
        	
    toothbrush = toothbrush_cascade.detectMultiScale(gray,1.3, 5)
    for (tx,ty,tw,th) in toothbrush:
        cv2.rectangle(roi_color,(tx,ty),(tx+tw,ty+th),(0,255,0),2)
        font=cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,word8,(tx-tw,ty+th),font,1,(255,255,0),2,cv2.LINE_AA)
        engine.say(word8)
        engine.setProperty('rate',100)
        engine.runAndWait()
        
        
    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

#engine = pyttsx3.init()
