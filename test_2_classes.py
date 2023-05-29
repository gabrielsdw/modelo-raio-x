import numpy as np
import tensorflow as tf
from PIL import Image
from utils import predictList


def predicty(image_path, show=False):
    image = Image.open(image_path).resize((64, 64))
    if show:
        image.show()
    
    image_array = np.asarray(image)
    image_array = image_array / 255
    
    image_array = np.expand_dims(image_array, axis=0)
    
    model = tf.keras.models.load_model("checkpoints/teste7.h5")
    
    prediction = model.predict(image_array)
    print(prediction)

    predict_dict = {
        0: "Anormal",
        1: "Normal"
    }

    if prediction >= 0.5:
        return predict_dict[1]
    return predict_dict[0]


result = predictList(function=predicty, directory="imagens_teste_anormalidade")
