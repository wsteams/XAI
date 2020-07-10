import cntk as C
import cv2
import numpy as np

img_channel = 3
img_height = 227
img_width = 227
img_mean = 114


class AdversarialImage:
    def __init__(self, model):
        self.model = model

    def fast_method(self, img, epsilon=8):
        pred = int(np.argmax(self.model.eval({self.model.arguments[0]: img})))
        grad = self.model[pred, :].grad({self.model.arguments[0]: img})[0]
        adversarial = img + epsilon * np.sign(grad)
        return np.clip(adversarial / 255, 0, 1) * 255

    def iterative_fast(self, img, epsilon=8, alpha=1):
        adversarial = img.copy()

        pred = int(np.argmax(self.model.eval({self.model.arguments[0]: img})))
        for _ in range(int(min(epsilon + 4, 1.25 * epsilon))):
            grad = alpha * np.sign(self.model[pred, :].grad({self.model.arguments[0]: adversarial})[0])
            adversarial = np.clip(adversarial + grad, img - epsilon, img + epsilon)
        return np.clip(adversarial / 255, 0, 1) * 255

    def least_likely(self, img, epsilon=8, alpha=1):
        adversarial = img.copy()

        pred = int(np.argmin(self.model.eval({self.model.arguments[0]: img})))
        for _ in range(int(min(epsilon + 4, 1.25 * epsilon))):
            grad = alpha * np.sign(self.model[pred, :].grad({self.model.arguments[0]: adversarial})[0])
            adversarial = np.clip(adversarial + grad, img - epsilon, img + epsilon)
        return np.clip(adversarial / 255, 0, 1) * 255


def LocalResponseNormalization(k, n, alpha, beta):
    x = C.placeholder()
    xs = C.reshape(C.square(x), (1, C.InferredDimension), 0, 1)
    W = C.constant(alpha / (2 * n + 1), (1, 2 * n + 1, 1, 1))
    y = C.convolution (W, xs)
    b = C.reshape(y, C.InferredDimension, 0, 2)
    return C.element_divide(x, C.exp(beta * C.log(k + b)))


def alexnet(h):
    model = C.load_model("./AlexNet_ImageNet_CNTK.model")

    params = model.parameters

    conv1 = C.relu(C.convolution(params[7].value, h, strides=4, auto_padding=[False, True, True]) + params[8].value)
    norm1 = LocalResponseNormalization(1.0, 2, 1e-4, 0.75)(conv1)
    pool1 = C.pooling(norm1, C.MAX_POOLING, pooling_window_shape=(3, 3), strides=(2, 2), auto_padding=[False, False, False])

    conv2 = C.relu(C.convolution(params[6].value, pool1, strides=1, auto_padding=[False, True, True]) + params[9].value)
    norm2 = LocalResponseNormalization(1.0, 2, 1e-4, 0.75)(conv2)
    pool2 = C.pooling(norm2, C.MAX_POOLING, pooling_window_shape=(3, 3), strides=(2, 2), auto_padding=[False, False, False])

    conv3 = C.relu(C.convolution(params[5].value, pool2, strides=1, auto_padding=[False, True, True]) + params[10].value)
    conv4 = C.relu(C.convolution(params[4].value, conv3, strides=1, auto_padding=[False, True, True]) + params[11].value)
    conv5 = C.relu(C.convolution(params[3].value, conv4, strides=1, auto_padding=[False, True, True]) + params[12].value)

    pool3 = C.pooling(conv5, C.MAX_POOLING, pooling_window_shape=(3, 3), strides=(2, 2), auto_padding=[False, False, False])

    h = C.relu(C.times(pool3, params[2].value.reshape(-1, 4096)) + params[13].value)
    h = C.relu(C.times(h, params[1].value) + params[14].value)
    h = C.times(h, params[0].value) + params[15].value

    return h


if __name__ == "__main__":
    input = C.input_variable(shape=(img_channel, img_height, img_width), dtype="float32", needs_gradient=True)
    
    model = alexnet(input)

    adversarial_image = AdversarialImage(model)

    img = cv2.resize(cv2.imread("./panda.jpg"), (img_width, img_height))

    x_img = np.ascontiguousarray(img.transpose(2, 0, 1), dtype="float32")

    #
    # adversarial images
    #
    x_fast = adversarial_image.fast_method(x_img)
    x_iterative = adversarial_image.iterative_fast(x_img)
    x_least = adversarial_image.least_likely(x_img)
    
