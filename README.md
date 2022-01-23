# Siamese-Network

### 1. Introduction

What is a Siamese network ? You can think it's a network contains 2 sub-networks and these 2 sub-network will share the same weights. The high level architecture is as below.

![highlevel](https://github.com/Qucy/Siamese-Network/blob/master/img/highlevel.jpg)

So what do we mean by sharing the same weights here ? You can think these 2 networks are actually the same networks. So no matter which network the input goes into, will have the same outputs. 

Many times when we want to compare the similarity of two images, we thinking that use neural network to extract the features and then do comparison. But if we use 2 different network to extract the features then we can not do an apple to apple comparison here. So that's why are sharing the weights between 2 networks in the Siamese network.

The Siamese network input 2 different image at a time, if the input images are the same category then label is 1 and if input images are belongs to different category then the label is 0. So we can use binary cross entropy loss here.

### 2. Network

In the original paper it's use VGG16 as backbone network. But in this source code we use ResNet18 + ECA attention block as our backbone network. And at the end VGG network outputs two 4096 vector and then calculate L1 loss between 2 vectors. In the end it adding another 2 fully connector layer, one is used to reduce the dimension and the other one with a sigmoid activation function to output similarity.

![network](https://github.com/Qucy/Siamese-Network/blob/master/img/network.jpg)

The whole network source code implemented via TF 2.7 is as below

```python
import os
import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential

os.environ['CPP_TF_MIN_LOG_LEVEL'] = '2'

print(tf.__version__)


class ResidualBlock(layers.Layer):
    """
    Residual block inputs -> CONV -> BatchNorm -> LeakyReLu -> CONV -> BatchNorm + concat + LeakyReLu
                      |______________________________________________________________|
    """
    def __init__(self, filter_num, strides=1):
        super(ResidualBlock, self).__init__()
        self.c1 = layers.Conv2D(filter_num, kernel_size=3, strides=strides, padding='same')
        self.b1 = layers.BatchNormalization()
        self.a1 = layers.LeakyReLU(.2)

        self.c2 = layers.Conv2D(filter_num, kernel_size=3, strides=1, padding='same')
        self.b2 = layers.BatchNormalization()

        if strides > 1:
            self.downSample = layers.Conv2D(filter_num, kernel_size=1, strides=strides)
        else:
            self.downSample = lambda x : x


    def call(self, inputs, *args, **kwargs):
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)
        x = self.c2(x)
        x = self.b2(x)
        shortcut = self.downSample(inputs)
        x = layers.add([x, shortcut])
        x = tf.nn.relu(x)
        return x


class ECABlock(layers.Layer):
    """
    ECA attention block
    Global Average pool + Global Max pool -> shared FC layers -> add -> sigmoid * inputs
    """
    def __init__(self, filter_nums, b=1, gamma=2):
        super(ECABlock, self).__init__()
        self.avg = layers.GlobalAveragePooling2D()
        self.kernel_size = int(abs((tf.math.log(tf.cast(filter_nums, dtype=tf.float32), 2) + b) / gamma))
        self.c1 = layers.Conv1D(1, kernel_size=self.kernel_size, padding="same", use_bias=False)


    def call(self, inputs, *args, **kwargs):
        attention = self.avg(inputs)
        attention = layers.Reshape((-1, 1))(attention)
        attention = self.c1(attention)
        attention = tf.nn.sigmoid(attention)
        attention = layers.Reshape((1, 1, -1))(attention)
        return layers.Multiply()([inputs, attention])


class AttentionResNet(Model):

    def __init__(self, filter_nums, layer_dims):
        """
        init function for Attention ResNet
        :param filter_nums: number of filter for each residual block and attention block
        :param layer_dims: [2,2,2,2] => how many residual blocks for each residual module
        :param num_classes: number of classes
        """
        super(AttentionResNet, self).__init__()
        self.input_layer = Sequential([
            layers.Conv2D(64, kernel_size=3, strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(.2),
            layers.MaxPool2D(pool_size=3, strides=1, padding='same')
        ])

        self.resBlock1 = self.buildResidualBlock(filter_nums[0], layer_dims[0])
        self.ecaBlock1 = ECABlock(filter_nums[0])

        self.resBlock2 = self.buildResidualBlock(filter_nums[1], layer_dims[1], strides=2)
        self.ecaBlock2 = ECABlock(filter_nums[1])

        self.resBlock3 = self.buildResidualBlock(filter_nums[2], layer_dims[2], strides=2)
        self.ecaBlock3 = ECABlock(filter_nums[2])

        self.resBlock4 = self.buildResidualBlock(filter_nums[3], layer_dims[3], strides=2)
        self.ecaBlock4 = ECABlock(filter_nums[3])

        # [b, 512, h, w] => [b, 512, 1, 1]
        self.avgPool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(512)


    def call(self, inputs, *args, **kwargs):
        x = self.input_layer(inputs)

        x = self.resBlock1(x)
        x = self.ecaBlock1(x)

        x = self.resBlock2(x)
        x = self.ecaBlock2(x)

        x = self.resBlock3(x)
        x = self.ecaBlock3(x)

        x = self.resBlock4(x)
        x = self.ecaBlock4(x)

        x = self.avgPool(x)
        x = self.fc(x)

        return x


    def buildResidualBlock(self, filter_number, blocks, strides=1):
        res_blocks = Sequential()
        # may down sample
        res_blocks.add(ResidualBlock(filter_number, strides))
        for _ in range(1, blocks):
            res_blocks.add(ResidualBlock(filter_number, strides=1))
        return res_blocks



class Siamese(Model):
    """
    Siamese network
    """
    def __init__(self):
        super(Siamese, self).__init__()
        self.network1 = AttentionResNet([64, 128, 256, 512], [2, 2, 2, 2])
        self.d1 = layers.Dense(512, activation=tf.nn.relu)
        self.d2 = layers.Dense(1, activation=tf.nn.sigmoid)


    def call(self, inputs, training=None, mask=None):
        outputs1 = self.network1(inputs[:,:32,:,:])
        outputs2 = self.network1(inputs[:,32:,:,:])
        l1 = tf.math.abs(outputs1 - outputs2)
        x = self.d1(l1)
        x = self.d2(x)
        return x

```

### 3.Training

##### 3.1 Training data

In this source code we use Omniglot as out datasets, it contains 1623 different hand written characters by 20 different Amazon. You can find this dataset in datasets folder and file named raw_data.zip. After you unzip the file you should the folder structure like below. In each character contains 20 images written by 20 different Amazon.  In the some character folder all refer to the same character. Some sampled data looks like below.

![ds_folder](https://github.com/Qucy/Siamese-Network/blob/master/img/ds_folder.jpg)

##### 3.2 Prepare training and labels

Before we generate TF datasets, we use pandas to create positive sample pairs and negative sample pairs. Because Siamese network take 2 inputs at time, so we need to pair 2 images and 1 label to create a single training data.

- positive samples, pair 2 two images in the **same** folder and with label **1**, in this source code we generated 10 pairs.
- negative samples, pair 2 two images in the **different** folder and with label **0**, in this source code we generated 10 pairs.

After pair images successfully, we concat paired images into 1 image and save to local disk. After this we can use TF function to build training datasets easily. For easy purpose in this source code, we just use jitter in data argumentation and we resize image from 105 x 105 to 32 x 32.

The whole source code for data prepare is as below

```python
import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

os.environ['CPP_TF_MIN_LOG_LEVEL'] = '2'

# folder path
raw_data_path = './datasets/raw_data'
positive_path = './datasets/positive/'
negative_path = './datasets/negative/'
# image dataframe path
image_path_df = './datasets/image_paths.csv'
positive_df_path = './datasets/pos.csv'
negative_df_path = './datasets/neg.csv'

# image height and width after jitter
IMG_HEIGHT, IMG_WIDTH = 105, 105

def load_image_and_labels():
    # image labels start from 0
    labels = 0
    image_paths = []
    image_labels = []

    ### loading all the image path and it's categorical into arrays
    ### looping by alphabet
    for alphabet in os.listdir(raw_data_path):
        alphabet_path = os.path.join(raw_data_path, alphabet)
        # looping by character
        for character in os.listdir(alphabet_path):
            character_path = os.path.join(alphabet_path, character)
            # looping image one by one
            for image in os.listdir(character_path):
                image_paths.append(os.path.join(character_path, image))
                image_labels.append(labels)
            labels += 1

    return image_paths, image_labels


def generate_positive_samples():
    if not os.path.isfile(positive_df_path):
        df = pd.read_csv(image_path_df)
        df.loc[:, 'image2'] = df['image'].shift(-1)
        df.rename(columns={'image':'image1'}, inplace=True)
        df = df.groupby('label').head(10).reset_index()
        df.drop(columns=['index'], inplace=True)
        df = df[['image1','image2','label']]
        df.to_csv(positive_df_path, index=False)


def generate_negative_samples():
    if not os.path.isfile(negative_df_path):
        df = pd.read_csv(image_path_df)
        df_shuffle = df.sample(frac=1, random_state=42).reset_index(drop=True)
        df.loc[:, 'image2'] = df_shuffle['image']
        df.loc[:, 'label2'] = df_shuffle['label']
        # select only not matched records
        df = df.loc[~(df['label'] == df['label2'])]
        # only select top 10 for each category
        df = df.groupby('label').head(10).reset_index()
        df.rename(columns={'image':'image1'}, inplace=True)
        df = df[['image1','image2','label']]
        df.to_csv(negative_df_path, index=False)


def resize_image(image1, image2, height, width):
    image1 = tf.image.resize(image1, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image2 = tf.image.resize(image2, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return image1, image2


def random_crop(image1, image2):
    # [150, 150, 3] => [2, 150, 150, 3]
    stacked_images = tf.stack([image1, image2], axis=0)
    cropped_images = tf.image.random_crop(stacked_images, [2, IMG_HEIGHT, IMG_WIDTH, 3])
    return cropped_images[0], cropped_images[1]


@tf.function
def random_jitter(image1, image2):
    # resize image to 150 * 150
    image1, image2 = resize_image(image1, image2, 130, 130)
    # random crop image
    image1, image2 = random_crop(image1, image2)
    # resize to a smaller img for better local training
    image1, image2 = resize_image(image1, image2, 32, 32)
    # flip image
    if tf.random.uniform(()) > 0.5:
        image1 = tf.image.flip_left_right(image1)
        image2 = tf.image.flip_left_right(image2)

    return image1, image2



def concat_images_and_write_target_folder(target_file, target_folder):
    # if image already generated
    if len(os.listdir(target_folder)) == 16230:
        return
    samples = pd.read_csv(target_file)
    idx = 1
    for path1, path2 in zip(samples['image1'].values, samples['image2']):
        img1 = cv2.imread(path1)
        img2 = cv2.imread(path2)
        img1, img2 = random_jitter(img1, img2)
        concat_img = np.concatenate((img1, img2), axis=0)
        cv2.imwrite(os.path.join(target_folder, f'{idx}.jpg'), concat_img)
        idx += 1


if __name__ == '__main__':

    positive_df = pd.DataFrame()
    negative_df = pd.DataFrame()

    ### loading image path and labels, save into a csv file
    if not os.path.isfile(image_path_df):

        image_paths, image_labels = load_image_and_labels()

        image_file_df = pd.DataFrame(data={'image':image_paths, 'label':image_labels})

        image_file_df.to_csv(image_path_df, index=False)

    ### generate positive samples and negative samples
    ### because every character folder contains 20 images which belongs to same category
    ### so for each character folder we're going to generate 10 positive samples and 10 negative samples
    ### here positive samples means image from the same character folder
    ### while negative samples means image from different character folder
    generate_positive_samples()
    generate_negative_samples()
    concat_images_and_write_target_folder(positive_df_path, positive_path)
    concat_images_and_write_target_folder(negative_df_path, negative_path)
```

##### 3.3 Train and predict

After data is prepared you can use train.py and predict.py to train or make predictions. Below are some prediction results from my local PC after training for 4 epochs. Row 1 is the prediction result for positive samples and row 2 is prediction result for negative samples.

![results](https://github.com/Qucy/Siamese-Network/blob/master/img/results.jpg)

![results](https://github.com/Qucy/Siamese-Network/blob/master/img/results1.jpg)

Besides Omniglot datasets, I've also tried on some random hand written signatures and returned expected results. This would be another option model for compare signature similarity.

![results](https://github.com/Qucy/Siamese-Network/blob/master/img/signature_compare.jpg)



 
