import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Identity, LeakyReLU, UpSampling2D

from einops import rearrange

class CNNBlock(tf.keras.Layer):
    def __init__(self, out_channels, kernel_size=3, strides=1, padding='valid', use_bn=True):
        super().__init__()
        self.net=tf.keras.Sequential([
            Conv2D(out_channels,kernel_size=kernel_size, strides=strides, padding=padding, use_bias=not use_bn),
            BatchNormalization() if use_bn else Identity(),
            LeakyReLU(negative_slope=0.1) if use_bn else Identity(),
        ])

    def call(self,x):
        return self.net(x)

class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super(ResidualBlock, self).__init__()
        self.layers = []
        for _ in range(num_repeats):
            block = tf.keras.Sequential([
                Conv2D(channels//2, kernel_size=1, padding='same'),
                BatchNormalization(),
                LeakyReLU(negative_slope=0.1),
                Conv2D(channels, kernel_size=3, padding='same'),
                BatchNormalization(),
                LeakyReLU(negative_slope=0.1)
            ])
            self.layers.append(block)
        self.use_residual = use_residual
        self.num_repeats=num_repeats

    def call(self, x):
        for layer in self.layers:
            x = x + layer(x) if self.use_residual else layer(x)

        return x

class ScalePrediction(tf.keras.Layer):
    def __init__(self, channels, num_classes):
        super().__init__()
        self.net=tf.keras.Sequential([
            Conv2D(2*channels, kernel_size=3, padding='same'),
            BatchNormalization(),
            LeakyReLU(negative_slope=0.1),
            Conv2D((num_classes+5)*3, kernel_size=1)
        ])
        self.num_classes=num_classes

    def call(self, x):
        out=self.net(x)
        out=rearrange(out,'b w h (n_box n_feature) -> b n_box w h n_feature', n_feature=self.num_classes+5)
        return out

class YOLOv3(tf.keras.Model):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.layer_list=[
            CNNBlock(32, kernel_size=3, strides=1, padding='same'),
            CNNBlock(64, kernel_size=3, strides=2, padding='same'),
            ResidualBlock(64, num_repeats=1),

            CNNBlock(128, kernel_size=3, strides=2, padding='same'),
            ResidualBlock(128, num_repeats=1),

            CNNBlock(256, kernel_size=3, strides=2, padding='same'),
            ResidualBlock(256, num_repeats=2),

            CNNBlock(512, kernel_size=3, strides=2, padding='same'),
            ResidualBlock(512, num_repeats=2),

            CNNBlock(1024, kernel_size=3, strides=2, padding='same'),
            ResidualBlock(1024, num_repeats=1),

            CNNBlock(512, kernel_size=1, strides=1, padding='valid'),
            CNNBlock(1024, kernel_size=3, strides=1, padding='same'),
            ResidualBlock(1024, use_residual=False, num_repeats=1),

            CNNBlock(512, kernel_size=1, strides=1, padding='valid'),
            ScalePrediction(512, num_classes=num_classes),

            CNNBlock(256, kernel_size=1, strides=1, padding='valid'),
            UpSampling2D((2,2)),

            CNNBlock(256, kernel_size=1, strides=1, padding='valid'),
            CNNBlock(512, kernel_size=3, strides=1, padding='same'),
            ResidualBlock(512, use_residual=False, num_repeats=1),


            CNNBlock(256, kernel_size=1, strides=1, padding='valid'),
            ScalePrediction(256, num_classes=num_classes),

            CNNBlock(128, kernel_size=1, strides=1, padding='valid'),
            UpSampling2D((2,2)),

            CNNBlock(128, kernel_size=1, strides=1, padding='valid'),
            CNNBlock(256, kernel_size=3, strides=1, padding='same'),
            ResidualBlock(256, use_residual=False, num_repeats=1),

            CNNBlock(128, kernel_size=1, strides=1, padding='valid'),
            ScalePrediction(128, num_classes=num_classes)
        ]

    def call(self,x):
        outputs=[]
        route_connections=[]
        for i, layer in enumerate(self.layer_list):
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue
            x=layer(x)

            if isinstance(layer, ResidualBlock) and layer.num_repeats==2:
                route_connections.append(x)
            elif isinstance(layer, UpSampling2D):
                x=tf.concat([x, route_connections[-1]], axis=-1)
                route_connections.pop()

        return outputs

class YoloMimic(tf.keras.Model):
    def __init__(self, IN_CHANNELS, NUM_CLASSES, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = IN_CHANNELS
        self.num_classes = NUM_CLASSES

        self.net_1 = Conv2D((NUM_CLASSES+5)*3, kernel_size=3, strides=32, padding='same')
        self.net_2 = Conv2D((NUM_CLASSES+5)*3, kernel_size=3, strides=16, padding='same')
        self.net_3 = Conv2D((NUM_CLASSES+5)*3, kernel_size=3, strides=8, padding='same')

    def call(self, x):
        x1 = tf.reshape(self.net_1(x), [-1, 3, 7, 7, self.num_classes+5])
        x2 = tf.reshape(self.net_2(x), [-1, 3, 14, 14, self.num_classes+5])
        x3 = tf.reshape(self.net_3(x), [-1, 3, 28, 28, self.num_classes+5])

        return x1, x2, x3

    def get_config(self):
        config = super().get_config()
        config.update({
            'IN_CHANNELS': self.in_channels,
            'NUM_CLASSES': self.num_classes,
        })
        return config

    # @classmethod
    # def from_config(cls, config):
    #     return cls(**config)
