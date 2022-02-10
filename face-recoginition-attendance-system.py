import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
#images path
path = 'images'

#images name list
images = []

#names in images folder
classnames = []

#all images file in mylist folder
mylist = os.listdir(path)
print(mylist)

#printing all images name without extenstion
for cl in mylist:
    curimg = cv2.imread(f'{path}/{cl}')
    images.append(curimg)
    classnames.append(os.path.splitext(cl)[0])

print(classnames)

#function to find encodings of face within image
def findEncoding(images):
    encodelist =[]
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist

#marking attendence with in a file
def markAttendence(name):
    with open('attendence.csv','r+') as f:
        mydatalist = f.readlines()
        namelist =[]
        #print(mydatalist)
        for line in mydatalist:
            entry = line.split(',')
            namelist.append(entry[0])

        #print(namelist)
        if name not in namelist:
            now = datetime.now()
            dtstring = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtstring}')

#passing images list to find thier encodings
encodelistknown = findEncoding(images)
print('encoding complete')
#print(encodelistknown)

cap = cv2.VideoCapture(0)
print("enter Q to Quit")
while True:
    success,img = cap.read()
    imgs = cv2.resize(img,(0,0),None,0.25,0.25)
    imgs = cv2.cvtColor(imgs,cv2.COLOR_BGR2RGB)

    facecurframe = face_recognition.face_locations(imgs)
    encodecurframe = face_recognition.face_encodings(imgs,facecurframe)

    for encodeface,faceloc in zip(encodecurframe,facecurframe):
        matches = face_recognition.compare_faces(encodelistknown,encodeface)
        facedis = face_recognition.face_distance(encodelistknown,encodeface)
        print(facedis)
        matchindex = np.argmin(facedis)

        if(matches[matchindex]):
            name = classnames[matchindex].upper()
            y1,x2,y2,x1 = faceloc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1-20,y1-20),(x2+20,y2+20),(0,255,0),2)
            cv2.rectangle(img,(x1-20,y2-15),(x2+20,y2+20),(0,0,255),cv2.FILLED)
            cv2.putText(img,name,(x1+26,y2+18),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
            markAttendence(name)



    cv2.imshow('webcam',img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
