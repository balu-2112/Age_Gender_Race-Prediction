#importing necessary libraries
import cv2
import os
import numpy as np
from keras.models import load_model
import tensorflow as tf

#getting current working directory
path = os.getcwd()


#Details for the Image
IMG_WIDTH = IMG_HEIGHT = 200
Id_Gen = {0: 'male', 1: 'female'}
Gen_Id = dict((g, i) for i, g in Id_Gen.items())
Id_Race = {0: 'white', 1: 'black', 2: 'asian', 3: 'indian', 4: 'others'}
Race_Id = dict((r, i) for i, r in Id_Race.items())



print("Loading PreTrained Model......")

#Loading Pretrained Inception Model Multi-CNN
model_path = path + "\\" + 'Age_Gender_Race_Model.h5'
model = load_model(model_path)

#Loading haarcasecase Face Detector
file = path + "\\" + 'haarcascade_frontalface_default.xml'
print(file)

font = cv2.FONT_HERSHEY_SIMPLEX

face_cascade = cv2.CascadeClassifier(file)

#Reading Frames using Web Cam
print("Starting Web Cam....")
cap = cv2.VideoCapture(0)
print("Web Cam Started.....")
#Width
cap.set(3, 800) 
#Height
cap.set(4, 500) 

while True:
    #Capture frame-by-frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.5,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    #drawing rectangular box on detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

        roi = frame[y:y+h, x:x+w]
        #cv2.imwrite("roi.jpg", roi)
        roi = cv2.resize(roi, (200, 200))
        
        roi = np.array(roi) / 255.0

        roi = roi.reshape(-1,200,200,3)
        
        #getting the age, race and gender from the pretrained model
        age, race, gender = model.predict(roi)

        age, race, gender = [int((age[0]*100)/1.8),Id_Race[race.argmax()],Id_Gen[gender.argmax()]]
        
        text = "Age: " + str(age) + " Gender: " + str(gender) + " Race: " + str(race)

        cv2.putText(frame,text,(x,y-10), font, 0.2, (255, 0, 255), 1, cv2.LINE_AA)

    cv2.imshow('Age_Gender_Detector', frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
