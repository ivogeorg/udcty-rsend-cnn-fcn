# Udacity Robotics Software Engineer Nanodegree

## Semantic segmentation for object recognition

### Fully Convolutional Networks (FCN)

Introduced by Long and Shelhamer (Berkeley) | [CVPR'15](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf) | [Github](https://github.com/shelhamer/fcn.berkeleyvision.org).

### File `dense_to_1x1`

The correct use is t`f.layers.conv2d(x, num_outputs, 1, 1, weights_initializer=custom_init)`.

    * `num_outputs` defines the number of output channels or kernels
    * The third argument is the kernel size, which is 1.
    * The fourth argument is the stride, we set this to 1.
    * Use a custom initializer so the weights in the dense and convolutional layers are identical.

This results in the a _matrix multiplication_ operation that _preserves spatial information_.


