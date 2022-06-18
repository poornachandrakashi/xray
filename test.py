import os
from imutils import paths
import cv2
import numpy as np
import tensorflow as tf

def predict_result(images) :
    list1 = []

    for image_path in images:
        image = cv2.imread(image_path)
        model = tf.keras.models.load_model('/Users/adithya/IdeaProjects/xray/covid19.model')

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        input = np.array([image])

        (infected, normal) = model.predict(input)[0]

        if (infected > normal) :
            list1.append(("INFECTED", infected))
        else:
            list1.append(("NORMAL", normal))
    return np.array(list1)


imagePaths = list(paths.list_images('dataset/covid'))
print(predict_result(imagePaths))

