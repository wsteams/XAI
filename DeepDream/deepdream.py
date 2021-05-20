import cntk as C
import cv2
import matplotlib.pyplot as plt
import numpy as np

num_octave = 4
octave_scale = 1.4
iterations = 100
alpha = 1.5
epsilon = 1e-7

is_write = False


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


def inception_module(h, W1x1, b1x1, W3x3, b3x3, W3x3r, b3x3r, W5x5, b5x5, W5x5r, b5x5r, Wmax, bmax, name=''):
    """ Inception module GoogLeNet

    Architecture
    ------------
                  previous
                     |
       -----------------------------
       |        |         |        |
       |     conv1x1   conv1x1   max3x3
    conv1x1     |         |        |
       |     conv3x3   conv5x5  conv1x1
       |        |         |        |
       -----------------------------
                     |
                   concat
    """
    # 1x1 convolution
    conv1x1 = C.relu(convolution(W1x1, b1x1, name=name + "_1x1")(h))

    # 1x1 reduction -> 3x3 convolution
    conv3x3 = C.relu(convolution(W3x3r, b3x3r, name=name + "_3x3_bottleneck")(h))
    conv3x3 = C.relu(convolution(W3x3, b3x3, name=name + "_3x3")(conv3x3))

    # 1x1 reduction -> 5x5 convolution
    conv5x5 = C.relu(convolution(W5x5r, b5x5r, name=name + "_5x5_bottleneck")(h))
    conv5x5 = C.relu(convolution(W5x5, b5x5, name=name + "_5x5")(conv5x5))

    # 3x3 max pool -> 1x1 reduction
    pool3x3 = C.relu(convolution(Wmax, bmax, name=name + "_pool")(C.layers.MaxPooling((3, 3), strides=1, pad=True)(h)))

    return C.splice(conv1x1, conv3x3, conv5x5, pool3x3, axis=0, name=name)


def create_googlenet(h):
    """
    https://www.cntk.ai/Models/Caffe_Converted/BNInception_ImageNet_Caffe.model
    """
    model = C.load_model("./BNInception_ImageNet_Caffe.model")

    params = model.parameters

    conv1 = C.relu(convolution(params[24].value, params[25].value, stride=2, name="conv1")(h))
    pool1 = C.layers.MaxPooling((3, 3), strides=2, pad=True)(conv1)
    norm1 = local_response_normalization(1.0, 2, 1e-4, 0.75)(pool1)

    conv2 = C.relu(convolution(params[22].value, params[23].value, pad=False, name="conv2_1")(norm1))
    conv2 = C.relu(convolution(params[20].value, params[21].value, name="conv2_2")(conv2))
    norm2 = local_response_normalization(1.0, 2, 1e-4, 0.75)(conv2)

    pool2 = C.layers.MaxPooling((3, 3), strides=2, pad=True)(norm2)

    icp3a = inception_module(pool2, params[18].value, params[19].value, params[26].value, params[27].value,
                             params[28].value, params[29].value, params[30].value, params[31].value,
                             params[32].value, params[33].value, params[34].value, params[35].value, name="icp3a")
    icp3b = inception_module(icp3a, params[16].value, params[17].value, params[36].value, params[37].value,
                             params[38].value, params[39].value, params[40].value, params[41].value,
                             params[42].value, params[43].value, params[44].value, params[45].value, name="icp3b")

    pool3 = C.layers.MaxPooling((3, 3), strides=2, pad=True)(icp3b)

    icp4a = inception_module(pool3, params[14].value, params[15].value, params[46].value, params[47].value,
                             params[48].value, params[49].value, params[50].value, params[51].value,
                             params[52].value, params[53].value, params[54].value, params[55].value, name="icp4a")

    icp4b = inception_module(icp4a, params[12].value, params[13].value, params[56].value, params[57].value,
                             params[58].value, params[59].value, params[60].value, params[61].value,
                             params[62].value, params[63].value, params[64].value, params[65].value, name="icp4b")

    icp4c = inception_module(icp4b, params[10].value, params[11].value, params[66].value, params[67].value,
                             params[68].value, params[69].value, params[70].value, params[71].value,
                             params[72].value, params[73].value, params[74].value, params[75].value, name="icp4c")

    icp4d = inception_module(icp4c, params[8].value, params[9], params[76].value, params[77].value,
                             params[78].value, params[79].value, params[80].value, params[81].value,
                             params[82].value, params[83].value, params[84].value, params[85].value, name="icp4d")

    icp4e = inception_module(icp4d, params[6].value, params[7].value, params[86].value, params[87].value,
                             params[88].value, params[89].value, params[90].value, params[91].value,
                             params[92].value, params[93].value, params[94].value, params[95].value, name="icp4e")

    pool4 = C.layers.MaxPooling((3, 3), strides=2, pad=True)(icp4e)

    icp5a = inception_module(pool4, params[4].value, params[5].value, params[96].value, params[97].value,
                             params[98].value, params[99].value, params[100].value, params[101].value,
                             params[102].value, params[103].value, params[104].value, params[105].value, name="icp5a")

    icp5b = inception_module(icp5a, params[2].value, params[3].value, params[106].value, params[107].value,
                             params[108].value, params[109].value, params[110].value, params[111].value,
                             params[112].value, params[113].value, params[114].value, params[115].value, name="icp5b")

    pool5 = C.layers.GlobalAveragePooling()(icp5b)

    fc = dense(params[0].value, params[1].value, name="fc")(pool5)

    return fc


if __name__ == "__main__":
    img0 = cv2.imread("./sample.png")

    #
    # deep dream
    #
    octave_list = []

    img = img0.copy().astype("float32")
    for i in range(num_octave - 1):
        h, w = img.shape[:2]
        low = cv2.resize(img, (int(w / octave_scale), int(h / octave_scale)), interpolation=cv2.INTER_LINEAR)
        high = img - cv2.resize(low, (w, h))
        img = low
        octave_list.append(high)

    #
    # rendering octaves
    #
    for octave in range(num_octave):
        if octave > 0:
            high = octave_list[-octave]
            h, w = high.shape[:2]
            img = cv2.resize(img, (w, h)) + high

        print(img.shape)

        img_h, img_w = img.shape[:2]
        input_sub = C.input_variable(shape=(3, img_h, img_w), needs_gradient=True)
        model = create_googlenet(input_sub)

        model_sub = C.combine([model.icp5b_3x3])

        for i in range(iterations):
            """ Tile-method calculation """
            h, w = img.shape[:2]
            sz = max(h, w)
            sx, sy = np.random.randint(sz, size=2)
            img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
            grad = np.zeros_like(img)
            for y in range(0, max(h - sz // 2, sz), sz):
                for x in range(0, max(w - sz // 2, sz), sz):
                    sub = img_shift[y:y+sz, x:x+sz]
                    
                    g = C.reduce_mean(model_sub[200]).grad(
                        {input_sub: np.ascontiguousarray(sub.transpose(2, 0, 1), dtype="float32")})[0]

                    grad[y:y+sz, x:x+sz] = g.transpose(1, 2, 0)
                    
            g_roll = np.roll(np.roll(grad, -sx, 1), -sy, 0)
            img += g_roll * (alpha / np.abs(g_roll).mean() + epsilon)

    nightmare = np.uint8(np.clip(img / 255.0, 0, 1) * 255)

    plt.figure()
    plt.imshow(nightmare[..., ::-1])
    plt.axis("off")
    plt.show()
    
