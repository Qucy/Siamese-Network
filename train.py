import tensorflow as tf
from model import Siamese

NEGATIVE_LABEL = 0
POSITIVE_LABEL = 1
BATCH_SIZE =128
lr = 1e-4
model_weights = './datasets/model/siamese.tf'


def process_positive_ds(file_path):
    raw = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(raw, channels=3)
    return image, POSITIVE_LABEL

def process_negative_ds(file_path):
    raw = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(raw, channels=3)
    return image, NEGATIVE_LABEL


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    return x, y


# loading positive samples
positive_ds = tf.data.Dataset.list_files('./datasets/positive/*jpg')
positive_ds = positive_ds.map(process_positive_ds)
# loading negative samples
negative_ds = tf.data.Dataset.list_files('./datasets/negative/*jpg')
negative_ds = negative_ds.map(process_negative_ds)
# concat 2 datasets
train_ds = positive_ds.concatenate(negative_ds)
# preprocess
train_ds = train_ds.map(preprocess).cache().shuffle(35000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# construct model
siamese = Siamese()
# compile model
siamese.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.01),
                metrics=['accuracy'])

# fit
siamese.fit(train_ds, epochs=4)

# save weights
siamese.save_weights(model_weights, overwrite=True, save_format='tf')

