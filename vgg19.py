import tensorflow as tf


class VGG19Net(tf.keras.Model):

    def get_conv_layer(self, num_filters):
        xav_init = tf.contrib.layers.xavier_initializer
        return tf.layers.Conv2D(
            filters=num_filters,
            kernel_size=[3, 3],
            strides=1,
            padding="same",
            activation=tf.nn.relu,
            kernel_initializer=xav_init())

    def __init__(self, num_classes):
        super(VGG19Net, self).__init__()
        self.num_classes = num_classes

        self.layer_1 = self.get_conv_layer(num_filters=64)
        self.layer_2 = self.get_conv_layer(num_filters=64)

        self.layer_3 = self.get_conv_layer(num_filters=128)
        self.layer_4 = self.get_conv_layer(num_filters=128)

        self.layer_5 = self.get_conv_layer(num_filters=256)
        self.layer_6 = self.get_conv_layer(num_filters=256)
        self.layer_7 = self.get_conv_layer(num_filters=256)
        self.layer_8 = self.get_conv_layer(num_filters=256)

        self.layer_9 = self.get_conv_layer(num_filters=512)
        self.layer_10 = self.get_conv_layer(num_filters=512)
        self.layer_11 = self.get_conv_layer(num_filters=512)
        self.layer_12 = self.get_conv_layer(num_filters=512)

        self.layer_13 = self.get_conv_layer(num_filters=512)
        self.layer_14 = self.get_conv_layer(num_filters=512)
        self.layer_15 = self.get_conv_layer(num_filters=512)
        self.layer_16 = self.get_conv_layer(num_filters=512)

        self.layer_17 = tf.layers.Dense(units=4096, activation=tf.nn.relu)
        self.layer_18 = tf.layers.Dense(units=4096, activation=tf.nn.relu)
        self.layer_19 = tf.layers.Dense(
            units=self.num_classes, activation=tf.nn.relu)

    def call(self, inputs):
        x = tf.convert_to_tensor(inputs)

        # maxpooling is done over 2x2 windows, with stride 2
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = tf.layers.max_pooling2d(x, pool_size=[2, 2], strides=[2, 2])

        x = self.layer_3(x)
        x = self.layer_4(x)
        x = tf.layers.max_pooling2d(x, pool_size=[2, 2], strides=[2, 2])

        x = self.layer_5(x)
        x = self.layer_6(x)
        x = self.layer_7(x)
        x = self.layer_8(x)
        x = tf.layers.max_pooling2d(x, pool_size=[2, 2], strides=[2, 2])

        x = self.layer_9(x)
        x = self.layer_10(x)
        x = self.layer_11(x)
        x = self.layer_12(x)
        x = tf.layers.max_pooling2d(x, pool_size=[2, 2], strides=[2, 2])

        x = self.layer_13(x)
        x = self.layer_14(x)
        x = self.layer_15(x)
        x = self.layer_16(x)
        x = tf.layers.max_pooling2d(x, pool_size=[2, 2], strides=[2, 2])

        # FC Layers
        x = self.layer_17(x)
        x = self.layer_18(x)
        x = self.layer_19(x)

        # x is the logits now
        return x
