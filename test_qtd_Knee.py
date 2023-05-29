import numpy as np
import tensorflow as tf
from PIL import Image
from utils import predictList

def predictQuantityKnee(img_path):
    model = tf.keras.models.load_model("checkpoints/melhor_modelo_06.h5")

    img_show = Image.open(img_path)

    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(64, 64))

    img_array = tf.keras.preprocessing.image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)

    img_array = img_array / 255.0

    prediction = model.predict(img_array)
    prediction_label = np.argmax(prediction, axis=1)
    predict_dict = {
        0: 'One Knee',
        1: 'Photo proper knee',
        2: 'Two Knee'
    }

    return predict_dict[prediction_label[0]]


result = predictList(function=predictQuantityKnee, directory="imagens_teste_qtdKnee")
