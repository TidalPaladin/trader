"""
This module provides a Tensorflow 2.0 implementation of a vision
classification network for Tiny ImageNet.
"""
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model

class Tail(tf.keras.Model):

    def __init__(self, out_width=32):
        """
        Arguments:
            out_width: Number of output feature maps. Default 32.
        """
        super().__init__()

        # Construct 7x7/2 convolution layer
        # No BN / ReLU, handled in later blocks
        self.conv = layers.Conv1D(
                filters=out_width,
                name='Tail_conv',
                kernel_size=3,
                strides=1,
                padding='same',
                use_bias=False,
                activation=None,
        )

    def call(self, inputs, training=False, **kwargs):
        """
        Runs the forward pass for this layer

        Arguments:
            input: input tensor(s)
            training: boolean, whether or not

        Keyword Arguments:
            Forwarded to call() of each component layer.

        Return:
            Output of forward pass
        """
        return self.conv(inputs, **kwargs)

class Bottleneck(tf.keras.Model):
    """
    Resnet style residual bottleneck block consisting of:
        1. 1x1/1 channel convolution, Ni / 4 bottleneck
        2. 3x3/1 spatial convolution
        3. 1x1/1 channel convolution, exit width bottleneck
    """

    def __init__(self, out_width, bottleneck=4):
        """
        Constructs a bottleneck block with the final number of output
        feature maps given by `out_width`. Bottlenecked layers will have
        output feature map count given by `out_width // bottleneck`.

        Arguments:
            out_width: Positive integer, number of output feature maps.

            bottleneck: Positive integer, factor by which to bottleneck
                        relative to `out_width`. Default 4.
        """
        super().__init__()

        # 1x1 depthwise convolution, enter the bottleneck
        self.channel_conv_1 = layers.Conv1D(
                filters=out_width // bottleneck,
                name='Bottleneck_enter',
                kernel_size=1,
                strides=1,
                use_bias=False,
                activation=None,
        )
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()

        # 3x3 depthwise separable convolution
        self.spatial_conv = layers.Conv1D(
                filters=out_width // bottleneck,
                name='Bottleneck_spatial',
                kernel_size=3,
                strides=1,
                use_bias=False,
                activation=None,
                padding='same'
        )
        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.ReLU()

        # 1x1 depthwise convolution, exit the bottleneck
        self.channel_conv_2 = layers.Conv1D(
                filters=out_width,
                name='Bottleneck_exit',
                kernel_size=1,
                strides=1,
                use_bias=False,
                activation=None,
        )
        self.bn3 = layers.BatchNormalization()
        self.relu3 = layers.ReLU()

        # Merge operation to join residual + main paths
        self.merge = layers.Add()

    def call(self, inputs, training=False, **kwargs):
        """
        Runs the forward pass for this layer

        Arguments:
            input: input tensor(s)
            training: boolean, whether or not

        Keyword Arguments:
            Forwarded to call() of each component layer.

        Return:
            Output of forward pass
        """

        # Enter bottleneck, depthwise convolution
        _ = self.bn1(inputs, training=training)
        _ = self.relu1(_)
        _ = self.channel_conv_1(_)

        # Spatial convolution, depthwise separable
        _ = self.bn2(_, training=training)
        _ = self.relu2(_)
        _ = self.spatial_conv(_)

        # Exit bottleneck, depthwise convolution
        _ = self.bn3(_, training=training)
        _ = self.relu3(_)
        _ = self.channel_conv_2(_)

        # Combine residual and main paths
        return self.merge([inputs, _], **kwargs)

class Downsample(tf.keras.Model):
    """
    Resnet style residual bottleneck block consisting of:
        1. 1x1/1 depthwise convolution + BN + ReLU (bottlenecked)
        2. 3x3/1 depthwise separable convolution + BN + ReLU (bottlenecked)
        3. 1x1/1 depthwise convolution + BN + ReLU
    """

    def __init__(self, out_width, bottleneck=4, stride=2):
        """
        Constructs a downsample block with the final number of output
        feature maps given by `out_width`. Stride of the spatial convolution
        layer is given by `stride`. Take care to increase width appropriately
        for a given spatial downsample.

        The first two convolutions are bottlenecked according to `bottleneck`.

        Arguments:
            out_width:  Positive integer, number of output feature maps.

            bottleneck: Positive integer, factor by which to bottleneck
                        relative to `out_width`. Default 4.

            stride:     Positive integer or tuple of positive integers giving
                        the stride of the depthwise separable convolution layer.
                        If a single value, row and col stride will be
                        set to the given value. If a tuple, assign row and
                        col stride from the tuple as (row, col).  Default 2.

        """
        super().__init__()

        # 1x1 convolution, enter the bottleneck
        self.channel_conv_1 = layers.Conv1D(
                filters=out_width // bottleneck,
                name='Downsample_enter',
                kernel_size=1,
                strides=1,
                use_bias=False,
                activation=None,
        )
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()

        # 3x3 depthwise separable spatial convolution
        self.spatial_conv = layers.Conv1D(
                filters=out_width // bottleneck,
                name='Downsample_conv',
                kernel_size=3,
                strides=stride,
                use_bias=False,
                activation=None,
                padding='same'
        )
        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.ReLU()

        # 1x1 convolution, exit the bottleneck
        self.channel_conv_2 = layers.Conv1D(
                filters=out_width,
                name='Downsample_exit',
                kernel_size=1,
                strides=1,
                use_bias=False,
                activation=None,
        )
        self.bn3 = layers.BatchNormalization()
        self.relu3 = layers.ReLU()

        # 3x3/2 convolution along main path
        self.main = layers.Conv1D(
                filters=out_width,
                name='Downsample_main',
                kernel_size=3,
                strides=stride,
                use_bias=False,
                activation=None,
                padding='same'
        )
        self.bn_main = layers.BatchNormalization()
        self.relu_main = layers.ReLU()

        # Merge operation to join residual + main paths
        self.merge = layers.Add()

    def call(self, inputs, training=False, **kwargs):
        """
        Runs the forward pass for this layer

        Arguments:
            input: input tensor(s)
            training: boolean, whether or not

        Keyword Arguments:
            Forwarded to call() of each component layer.

        Return:
            Output of forward pass
        """

        # Enter bottleneck
        _ = self.bn1(inputs, training=training)
        _ = self.relu1(_)
        _ = self.channel_conv_1(_)

        # Spatial convolution
        _ = self.bn2(_, training=training)
        _ = self.relu2(_)
        _ = self.spatial_conv(_)

        # Exit bottleneck
        _ = self.bn3(_, training=training)
        _ = self.relu3(_)
        _ = self.channel_conv_2(_)

        # Main path with convolution
        # TODO can we use first residual BN + ReLU here?
        m = self.bn_main(inputs, training=training)
        m = self.relu_main(m)
        main = self.main(m)

        # Combine residual and main paths
        return self.merge([main, _])


class Head(tf.keras.Model):
    """
    Basic vision classification network head consisting of:
        1. 1D Global average pooling
        2. Fully connected layer + BN + ReLU
    """

    def __init__(self, classes=100):
        """
        Arguments:
            classes:    Positive integer, number of classes in the output of the
                        fully connected layer.

        Keyword Arguments:
            Forwarded to the dense layer.
        """
        super().__init__()

        self.global_avg = layers.GlobalAveragePooling1D()
        self.dense = layers.Dense(
                units=classes,
                use_bias=True,
                activation=None,
                name='Head_dense',
        )
        self.softmax = layers.Softmax()

    def call(self, inputs, training=False, **kwargs):
        _ = self.global_avg(inputs)
        _ = self.dense(_)
        return _ if training else self.softmax(_)


class TinyImageNet(tf.keras.Model):

    def __init__(self, levels, use_head=True, use_tail=True):
        """
        Arguments:
            levels: List of positive integers. Each list entry denotes a level of
                    downsampling, with the value of the i'th entry giving the number
                    of times the bottleneck layer is repeated at the i;th level

            use_head: boolean, if true include a default network head
            use_tail: boolean, if true include a default network tail

        Keyword Arguments:
            Forwarded to tf.keras.Model
        """
        super().__init__()
        width = 32

        # Use default tail if requested in params
        self.tail = Tail(out_width=width) if use_tail else None

        # Loop through levels and their parameterized repeat counts
        self.blocks = list()
        for level, repeats in enumerate(levels):

            # Create `repeats` Bottleneck blocks and add to the block list
            for block in range(repeats):
                bottleneck_layer = Bottleneck(out_width=width)
                self.blocks.append(bottleneck_layer)

            # Create a downsample layer that doubles width
            # Default stride=2
            downsample_layer = Downsample(out_width=2*width)
            self.blocks.append(downsample_layer)

            # Update `width` for the next iteration
            width *= 2

        self.final_bn = layers.BatchNormalization(name='final_bn')
        self.final_relu = layers.ReLU(name='final_relu')

        # Use default head if requested in params
        self.head = Head(classes=5) if use_head else None

    def call(self, inputs, training=False, **kwargs):
        """
        Runs the forward pass for this layer

        Arguments:
            input: input tensor(s)
            training: boolean, whether or not

        Keyword Arguments:
            Forwarded to call() of each component layer.

        Return:
            Output of forward pass
        """
        _ = self.tail(inputs, training=training) if self.tail else inputs

        # Loop over layers by level
        for layer in self.blocks:
            _ = layer(_, training=training)

        # Finish up specials in level 2
        _ = self.final_bn(_, training=training)
        _ = self.final_relu(_)

        return self.head(_, training, **kwargs) if self.head else _
