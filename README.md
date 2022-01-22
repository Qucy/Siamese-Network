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

In this source code we use Omniglot as out datasets, it contains 1623 different hand written characters by 20 different Amazons. 

