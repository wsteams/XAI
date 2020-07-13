import cntk as C
import numpy as np

C.debugging.force_deterministic(0)

img_channel = 3
img_height = 224
img_width = 224
img_mean = np.array([[[104]], [[117]], [[124]]], dtype="float32")

alpha = 1.0
beta = 1.0
epsilon = 1e-5


def convolution(weights, bias, pad=True, stride=1, name=''):
    W = C.Constant(value=weights, name='W')
    b = C.Constant(value=bias, name='b')

    @C.BlockFunction('Convolution2D', name)
    def conv2d(x):
        return C.convolution(W, x, strides=[stride, stride], auto_padding=[False, pad, pad]) + b

    return conv2d


def dense(weights, bias, name=''):
    W = C.Constant(value=weights, name='W')
    b = C.Constant(value=bias, name='b')

    @C.BlockFunction('Dense', name)
    def fc(x):
        return C.times(x, W) + b

    return fc


def vgg19(h):
    """
    https://www.cntk.ai/Models/Caffe_Converted/VGG19_ImageNet_Caffe.model
    """
    model = C.load_model("./VGG19_ImageNet_Caffe.model")

    params = model.parameters
    for i in range(16):
        h = convolution(params[-(2 * i + 2)].value, params[-(2 * i + 1)].value)(h)
        h = C.relu(h)
        if i in [1, 3, 7, 11, 15]:
            h = C.layers.MaxPooling((2, 2), strides=2, pad=True)(h)

    h = C.relu(dense(params[4].value, params[5].value)(h))
    h = C.relu(dense(params[2].value, params[3].value)(h))
    h = dense(params[0].value, params[1].value)(h)

    return h


if __name__ == "__main__":
    #
    # ImageNet categories
    #
    node = 65  # sea snake

    #
    # input and model
    #
    input = C.input_variable(shape=(img_channel, img_height, img_width), dtype="float32", needs_gradient=True)
    
    model = vgg19(input - img_mean)

    img = np.ascontiguousarray((np.random.rand(img_channel, img_height, img_width) + img_mean, dtype="float32")

    #
    # activation maximization
    #
    activation = C.element_times(alpha, model[node])
    total_variation = C.reduce_sum(  # total variation regularization
        C.sqrt(C.square(input[:, 1:, :-1] - input[:, :-1, :-1]) + C.square(input[:, :-1, 1:] - input[:, :-1, :-1])))
    activation -= C.element_times(beta, (total_variation / np.prod(input.shape)))

    for i in range(300):
        grads = activation.grad({input: img})[0]
        grads /= (np.sqrt(np.mean(np.square(grads))) + epsilon)
        img += grads
    
