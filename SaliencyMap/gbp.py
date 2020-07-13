import cntk as C
import cv2
import numpy as np

from cntk import user_function, output_variable
from cntk.ops.functions import UserFunction

from saliency import divergence_map

C.debugging.force_deterministic(0)

img_channel = 3
img_height = 224
img_width = 224
img_mean = np.array([[[104]], [[117]], [[124]]], dtype="float32")


class GuidedReLU(UserFunction):
    def __init__(self, arg, name="GuidedReLU"):
        super(GuidedReLU, self).__init__([arg], name=name)

    def forward(self, argument, device=None, outputs_to_retain=None):
        relu_x = np.maximum(0, argument)
        self.relu_guide = (relu_x <= 0)
        return relu_x, relu_x

    def backward(self, state, root_gradients):
        relu_x = np.ones_like(state)
        relu_x[self.relu_guide] = 0
        return np.maximum(0, root_gradients) * relu_x

    def infer_outputs(self):
        return [output_variable(self.inputs[0].shape, self.inputs[0].dtype, self.inputs[0].dynamic_axes)]

    @staticmethod
    def deserialize(inputs, name, state):
        return GuidedReLU(inputs[0], name)

      
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


def create_vgg19(h):
    """
    https://www.cntk.ai/Models/Caffe_Converted/VGG19_ImageNet_Caffe.model
    """
    model = C.load_model("../ActMax/VGG19_ImageNet_Caffe.model")

    params = model.parameters
    for i in range(16):
        h = convolution(params[-(2 * i + 2)].value, params[-(2 * i + 1)].value, name="conv{}".format(i + 1))(h)
        h = user_function(GuidedReLU(h, name="relu{}".format(i + 1)))
        if i in [1, 3, 7, 11, 15]:
            h = C.layers.MaxPooling((2, 2), strides=2, pad=True)(h)

    h = user_function(GuidedReLU(dense(params[4].value, params[5].value)(h)))
    h = user_function(GuidedReLU(dense(params[2].value, params[3].value)(h)))
    h = dense(params[0].value, params[1].value)(h)

    return h


if __name__ == "__main__":
    #
    # input and model
    #
    input = C.input_variable(shape=(img_channel, img_height, img_width), dtype="float32", needs_gradient=True)
    vgg19 = create_vgg19(input)

    img = cv2.resize(cv2.imread("./cat.jpg"), (img_width, img_height))
    x_img = np.ascontiguousarray(img.transpose(2, 0, 1), dtype="float32")

    #
    # Guided Backpropagation
    #
    guided_backprop = C.combine([vgg19.relu16]).grad({input: x_img - img_mean})[0]
    guided_backprop = divergence_map(guided_backprop)
    
