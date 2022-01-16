import cv2
from tinydb import TinyDB

imageDir = 'images'
dataSetDir = 'dataset'
db = TinyDB('db.json')

cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)
faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faceRecognizer = cv2.face.LBPHFaceRecognizer_create()

faceRecognizer.read(dataSetDir + '/model.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

minWidth = 0.1 * cam.get(3)
minHeight = 0.1 * cam.get(4)

while True:
    retV, frame = cam.read()
    abu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetector.detectMultiScale(abu, 1.2, 5, minSize=(round(minWidth), round(minHeight)))
    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        id, confidence = faceRecognizer.predict(abu[y:y + h, x:x + h])

        confidenceText = " {0}%".format(round(100 - confidence))
        nameID = 'john do'
        if confidence <= 50:
            user = db.get(doc_id=id)
            nameID = user.get('name')

        cv2.putText(frame, str(nameID), (x + 5, y - 5), font, 1, (255, 255, 0), 2)
        cv2.putText(frame, str(confidenceText), (x + 5, y + h - 5), font, 1, (0, 255, 255), 2)

    cv2.imshow('cam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
