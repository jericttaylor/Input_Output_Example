# Import relevant libraries
import os
import matplotlib.pyplot as plt
import numpy as np
from keras.applications import inception_v3 as inc_net
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions

# Use a pretrained DNN (Inception V3) from the library above
inet_model = inc_net.InceptionV3()

# Define a pre-processing function that transforms any image file into a vector
# This function was copied from:
# https://github.com/marcotcr/lime/blob/master/doc/notebooks/Tutorial%20-%20Image%20Classification%20Keras.ipynb
def transform_img(path_list):
    out = []
    for img_path in path_list:
        img = image.load_img(img_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = inc_net.preprocess_input(x)
        out.append(x)
    return np.vstack(out)

# Return InceptionV3's top 5 decision for a given image and plot the image
# The image must be in a folder 'data' in the same directory as this script
images = transform_img([os.path.join('data','dogtaxes.jpg')])
plt.imshow(images[0] / 2 + 0.5)
preds = inet_model.predict(images)
for x in decode_predictions(preds)[0]:
    print(x)