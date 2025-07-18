import cv2
import os
import numpy as np

data_path = "faces"
labels = []
faces = []
label_id = {}

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

id = 0
for person in os.listdir(data_path):
    person_path = os.path.join(data_path, person)
    label_id[person] = id
    for image in os.listdir(person_path):
        img_path = os.path.join(person_path, image)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        faces.append(img)
        labels.append(id)
    id += 1

recognizer.train(faces, np.array(labels))
recognizer.save("trained_model.yml")

with open("labels.txt", "w") as f:
    for name, id in label_id.items():
        f.write(f"{id},{name}\n")