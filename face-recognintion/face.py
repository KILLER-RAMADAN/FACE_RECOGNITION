import face_recognition
import cv2
import numpy as np
import os
from datetime import datetime
import pyttsx3 as textspeach
engine=textspeach.init()
images_path="C:\\Users\\ahmed\\Desktop\\images2"
images_list=[]
names_without_extension=[]
mylist=os.listdir(images_path)
for name in mylist:
    selectimage=cv2.imread(f"{images_path}/{name}")
    images_list.append(selectimage)
    names_without_extension.append(os.path.splitext(name)[0])
def get_encoding(images_list):
    images_encoding_list=[]
    for img in images_list:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        encode=face_recognition.face_encodings(img)
        images_encoding_list.append(encode)
    return images_encoding_list
known_images=get_encoding(images_list)
print("Encoding Done...")
def attendance(name):
    with open("C:\\Users\\ahmed\\my projects\\face-recognintion\\GET INFORMATION\\attendance.csv",'r+') as f:
        my_attendance_list=f.readlines()
        list_of_names=[]
        lecture="M110"
        for line in my_attendance_list:
            check_face=line.split(",")
            list_of_names.append(check_face[0])
        if name not in list_of_names:
           #############data_time_by_hour_minute_seconds#############
            now=datetime.now()
            data_time_string=now.strftime("%H:%M:%S")
           #############data_time_by_days_month_years#############
            date =now.strftime('%d/%m/%Y')
            day,month,year=date.split("/")
            f.writelines(f"\n{name},{data_time_string},{date},{lecture}")
            statement=str(f"welcome {name}")
            engine.say(statement)
            engine.runAndWait()
        else:
            cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(255,0,0),cv2.FILLED)
            cv2.putText(img,'UNknown',(x1+6,y2-6),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),2)
            statement=str(f"UNKNON NAME")
            engine.say(statement)
            engine.runAndWait()
            cv2.destroyAllWindows()
camera=cv2.VideoCapture(0)
while True:
    success,img=camera.read() #_____read photo from cam____
    small__image=cv2.resize(img,(212,212),None,1.12,1.12)#___to make process speed we change photo size______
    small__image=cv2.cvtColor(small__image,cv2.COLOR_BGR2HSV)#_____convert image to rgp_____
    faceframe=face_recognition.face_locations(small__image)#____face_frame____
    faceencoding=face_recognition.face_encodings(small__image,faceframe)#__to get the Correct dimensions
    for encode_of_face ,location_of_face in zip(faceencoding,faceframe):# to matches with images_incoding_list
        face_matches=face_recognition.compare_faces(known_images,encode_of_face)#_____compare faces____
        distance_of_face=face_recognition.face_distance(known_images,encode_of_face)#_____compare faces with dimentions___
        match_index=np.argmin(distance_of_face)
        if face_matches[match_index]:
            name=names_without_extension[match_index].upper()
            y1,x2,y2,x1=location_of_face#_____to make a pox of face location____
            y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),2)
            attendance(name)
            
            
    cv2.imshow("camera",img)
    cv2.waitKey(3)
  