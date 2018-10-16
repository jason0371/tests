from mxnet import gluon as g

# Helpers
def _make_fire(squeeze_channels, expand1x1_channels, expand3x3_channels):
    out = g.nn.HybridSequential(prefix='')
    out.add(_make_fire_conv(squeeze_channels, 1))

    paths = g.contrib.nn.HybridConcurrent(axis=1, prefix='')
    paths.add(_make_fire_conv(expand1x1_channels, 1))
    paths.add(_make_fire_conv(expand3x3_channels, 3, 1))
    out.add(paths)

    return out

def _make_fire_conv(channels, kernel_size, padding=0):
    out = g.nn.HybridSequential(prefix='')
    out.add(g.nn.Conv2D(channels, kernel_size, padding=padding))
    out.add(g.nn.Activation('relu'))
    return out

# Net
class SqueezeNet(g.nn.HybridBlock):
    r"""SqueezeNet model from the `"SqueezeNet: AlexNet-level accuracy with 50x fewer parameters
    and <0.5MB model size" <https://arxiv.org/abs/1602.07360>`_ paper.
    SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.
    Parameters
    ----------
    version : str
        Version of squeezenet. Options are '1.0', '1.1'.
    classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self, pretrained=False, **kwargs):
        super(SqueezeNet, self).__init__(**kwargs)

        with self.name_scope():
            self.features = g.nn.HybridSequential(prefix='')

            self.features.add(g.nn.Conv2D(32, kernel_size=3, strides=2))
            self.features.add(g.nn.Activation('relu'))
            self.features.add(g.nn.MaxPool2D(pool_size=3, strides=2, ceil_mode=True))
            self.features.add(_make_fire(8, 32, 32))
            self.features.add(_make_fire(8, 32, 32))
            self.features.add(g.nn.MaxPool2D(pool_size=3, strides=2, ceil_mode=True))
            self.features.add(_make_fire(16, 64, 64))
            self.features.add(_make_fire(16, 64, 64))
            self.features.add(g.nn.MaxPool2D(pool_size=3, strides=2, ceil_mode=True))
            self.features.add(_make_fire(24, 96, 96))
            self.features.add(_make_fire(24, 96, 96))
            self.features.add(_make_fire(32, 128, 128))
            self.features.add(_make_fire(32, 128, 128))
            self.features.add(g.nn.Dropout(0.5))

            self.output = g.nn.HybridSequential(prefix='scene')

            if pretrained:
                 self.output.add(g.nn.Conv2D(200, kernel_size=1))
            else:
                self.output.add(g.nn.Conv2D(200, kernel_size=1))


            self.output.add(g.nn.Activation('relu'))
            self.output.add(g.nn.AvgPool2D(3))
            self.output.add(g.nn.Flatten())


    def hybrid_forward(self, F, x):
        x0 = self.features(x)
        x1 = self.output(x0)
        return x1

