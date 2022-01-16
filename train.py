import cv2
import numpy as np
import os
from PIL import Image

imageDir = 'images'
dataSetDir = 'dataset'


def getImagelabel(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSample = []
    faceIds = []
    for imagePath in imagePaths:
        PilImg = Image.open(imagePath).convert('L')  # Convert image to gray
        imgNum = np.array(PilImg, 'uint8')
        faceId = int(os.path.split(imagePath)[-1].split('_')[1])
        faces = faceDetector.detectMultiScale(imgNum)
        for (x, y, w, h) in faces:
            faceSample.append(imgNum[y:y + h, x:x + h])
            faceIds.append(faceId)
        return faceSample, faceId


faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

print('wait')
faces, ids = getImagelabel(imageDir)
faceRecognizer.train(faces, np.array(ids))

# save
faceRecognizer.write(dataSetDir + '/model.xml')
print('Train done')
