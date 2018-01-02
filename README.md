# Udacity Robotics Software Engineer Nanodegree

## Semantic segmentation for object recognition

### Fully Convolutional Networks (FCN)

Introduced by Long and Shelhamer (Berkeley) | [CVPR'15](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf) | [Github](https://github.com/shelhamer/fcn.berkeleyvision.org).

### File `dense_to_1x1.py`

The correct use is `tf.layers.conv2d(x, num_outputs, 1, 1, weights_initializer=custom_init)`.

  * `num_outputs` defines the number of output channels or kernels
  * The third argument is the kernel size, which is 1.
  * The fourth argument is the stride, we set this to 1.
  * Use a custom initializer so the weights in the dense and convolutional layers are identical.

This results in the a _matrix multiplication_ operation that _preserves spatial information_.

### File `upsample.py`

One possible answer is using [`tf.layers.conv2d_transpose(x, 3, (2, 2), (2, 2))`](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/conv2d_transpose) to upsample.

  * The second argument `3` is the number of kernels/output channels.
  * The third argument is the kernel size, `(2, 2)`. Note that the kernel size could also be `(1, 1)` and the output shape would be the same. However, if it were changed to `(3, 3)` note the shape would be `(9, 9)`, at least with `'VALID'` padding.
  * The fourth argument, the number of strides, is how we get from a height and width from `(4, 4)` to `(8, 8)`. If this were a regular convolution the output height and width would be `(2, 2)`.

### Skip connections

Skip connections are (elementwise) **additions** of pooling layers or convolutional layers (before nonlinearity) to the output of later layers. 

  * In an FCN decoder, this restores spatial detail. 
  * Skip connections are used extensively in [residual networks](https://arxiv.org/abs/1512.03385).

#### Implementing skip connections in Keras:
```python
from keras.layers import Conv2D, Input

# input tensor for a 3-channel 256x256 image
x = Input(shape=(256, 256, 3))
# 3x3 conv with 3 output channels (same as input channels)
y = Conv2D(3, (3, 3), padding='same')(x)
# this returns x + y.
z = keras.layers.add([x, y])
```
Keras [API Guide](https://keras.io/getting-started/functional-api-guide/).

#### Implementing skip connections in TensorFlow

The [slim](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim) package of TensorFlow. A nice example [here](https://chatbotslife.com/resnets-highwaynets-and-densenets-oh-my-9bb15918ee32).

