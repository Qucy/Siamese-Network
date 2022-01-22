import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from model import Siamese

os.environ['CPP_TF_MIN_LOG_LEVEL'] = '2'

model_weights = './datasets/model/siamese.tf'
positive_image_path = './datasets/positive/'
negative_image_path = './datasets/negative/'
num_of_image_to_check = 4


def load_image(path):
    return tf.image.decode_jpeg(tf.io.read_file(path), channels=3)


def preprocess(x):
    x = tf.cast(x, dtype=tf.float32) / 255.
    return x

# construct model
siamese = Siamese()
# load weights
siamese.load_weights(model_weights)
# retrieve number of images in positive and negative folder
num_of_positive_images = len(os.listdir(positive_image_path))
num_of_negative_images = len(os.listdir(negative_image_path))

# randomly pick num_of_image_to_check
image_idx = np.random.randint(num_of_positive_images+1, size=num_of_image_to_check)
pos_images, neg_images = [], []
inputs = None
for idx in image_idx:
    pos_image = load_image(os.path.join(positive_image_path, f'{idx}.jpg'))
    neg_image = load_image(os.path.join(negative_image_path, f'{idx}.jpg'))
    pos_images.append(pos_image)
    neg_images.append(neg_image)
    # stack image for prediction
    if inputs is None:
        inputs = tf.stack([pos_image, neg_image], axis=0)
    else:
        _inputs = tf.stack([pos_image, neg_image], axis=0)
        inputs = tf.concat([inputs, _inputs], axis=0)

# preprocess image and make prediction
inputs = preprocess(inputs)
# make prediction
prediction = siamese.predict(inputs)

print(prediction)

# plot image
f, axarr = plt.subplots(2, num_of_image_to_check, figsize=(12, 12))
p_idx = 0
for x in range(num_of_image_to_check):
    axarr[0,x].imshow(pos_images[x])
    axarr[0,x].set_title('Similarity score {:.3f}'.format(np.round(prediction[p_idx][0], 3)))
    axarr[0,x].set_xticklabels([])
    axarr[0,x].set_yticklabels([])
    p_idx +=1

    axarr[1,x].imshow(neg_images[x])
    axarr[1,x].set_title('Similarity score {:.3f}'.format(np.round(prediction[p_idx][0], 3)))
    axarr[1,x].set_xticklabels([])
    axarr[1,x].set_yticklabels([])
    p_idx += 1

plt.show()

