import cntk as C
import numpy as np

img_channel = 3
img_height = 224
img_width = 224
img_mean = np.array([[[104]], [[117]], [[124]]], dtype="float32")

alpha = 1.0
beta = 1.0
epsilon = 1e-5


def vgg19(h):
    model = C.load_model("./VGG19_ImageNet_Caffe.model")

    params = model.parameters
    for i in range(16):
        h = C.convolution(params[-(2 * i + 2)].value, h, strides=1, auto_padding=[False, True, True]) + params[-(2 * i + 1)].value
        h = C.relu(h)
        if i in [1, 3, 7, 11, 15]:
            h = C.pooling(h, C.MAX_POOLING, pooling_window_shape=(2, 2), strides=(2, 2), auto_padding=[False, True, True])

    h = C.reshape(h, -1)
    h = C.relu(C.times(h, params[4].value.reshape(-1, 4096)) + params[5].value)
    h = C.relu(C.times(h, params[2].value) + params[3].value)
    h = C.times(h, params[0].value) + params[1].value
    return h


if __name__ == "__main__":
    input = C.input_variable(shape=(img_channel, img_height, img_width), dtype="float32", needs_gradient=True)
    model = vgg19(input - img_mean)

    img = np.ascontiguousarray((np.random.rand(img_channel, img_height, img_width) + img_mean, dtype="float32")

    node = 65  # sea snake

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
    
