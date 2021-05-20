import cntk as C
import cv2
import matplotlib.pyplot as plt
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


def local_response_normalization(k, n, alpha, beta):
    x = C.placeholder()
    xs = C.reshape(C.square(x), (1, C.InferredDimension), 0, 1)
    W = C.constant(alpha / (2 * n + 1), (1, 2 * n + 1, 1, 1))
    y = C.convolution(W, xs)
    b = C.reshape(y, C.InferredDimension, 0, 2)
    return C.element_divide(x, C.exp(beta * C.log(k + b)))


def alexnet(h):
    """
    https://www.cntk.ai/Models/CNTK_Pretrained/AlexNet_ImageNet_CNTK.model
    """
    model = C.load_model("./AlexNet_ImageNet_CNTK.model")

    params = model.parameters

    conv1 = C.relu(convolution(params[7].value, params[8].value, stride=4, pad=False)(h))
    norm1 = local_response_normalization(1.0, 2, 1e-4, 0.75)(conv1)
    pool1 = C.layers.MaxPooling((3, 3), strides=2, pad=False)(norm1)

    conv2 = C.relu(convolution(params[6].value, params[9].value)(pool1))
    norm2 = local_response_normalization(1.0, 2, 1e-4, 0.75)(conv2)
    pool2 = C.layers.MaxPooling((3, 3), strides=2, pad=False)(norm2)

    conv3 = C.relu(convolution(params[5].value, params[10].value)(pool2))
    conv4 = C.relu(convolution(params[4].value, params[11].value)(conv3))
    conv5 = C.relu(convolution(params[3].value, params[12].value)(conv4))

    pool3 = C.layers.MaxPooling((3, 3), strides=2, pad=False)(conv5)

    h = C.relu(dense(params[2].value, params[13].value)(pool3))
    h = C.relu(dense(params[1].value, params[14].value)(h))
    h = dense(params[0].value, params[15].value)(h)

    return h


if __name__ == "__main__":
    #
    # input and model
    #
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
    
    #
    # visualization
    #
    plt.figure(figsize=(16, 16))
    for i, x in enumerate(img_list):
        output = C.softmax(model).eval({model.arguments[0]: x - img_mean})
        
        plt.subplot(2, 2, i + 1)
        plt.imshow(x.transpose(1, 2, 0)[..., ::-1].astype("uint8"))
        plt.title("%s %.2f%%" % (category[output.argmax()][:-1], output.max() * 100))
        plt.axis("off")
    plt.tight_layout()
    plt.show()
    
