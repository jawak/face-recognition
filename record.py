import cv2
from tinydb import TinyDB

cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)
faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
db = TinyDB('db.json')

imageDir = 'images'
name = input('Masukan Nama: ')
faceID = db.insert({'name': name})

i = 1
while True:
    retV, frame = cam.read()
    abu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetector.detectMultiScale(abu, 1.3, 5)
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            if cv2.waitKey(1) & 0xFF == ord('c'):
                fileName = 'face_' + str(faceID) + '_' + str(i) + '.jpg'
                cv2.imwrite(imageDir + '/' + fileName, abu[y:y + h, x:x + w])
                i += 1

    cv2.imshow('webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif i > 20:
        break

cam.release()
cv2.destroyAllWindows()
