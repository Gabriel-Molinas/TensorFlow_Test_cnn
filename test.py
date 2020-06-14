import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

longitud, altura = 64, 64
modelo = './modelo/modelo.h5'
pesos_modelo = './modelo/pesos.h5'
classifier = load_model(modelo)
classifier.load_weights(pesos_modelo)


test_image = image.load_img('./dataset/single_prediction/cat_or_dog_2.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
print(result)
if result[0][0] == 1:
    print('Es un Perro!!!')
else:
    print('Es un Gato!!!')

