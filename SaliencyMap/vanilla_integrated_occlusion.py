import cntk as C
import cv2
import numpy as np

from saliency import VanillaGradients, IntegratedGradients, Occlusion
from saliency import grayscale_map

C.debugging.force_deterministic(0)

img_channel = 3
img_height = 224
img_width = 224
img_mean = np.array([[[104]], [[117]], [[124]]], dtype="float32")


if __name__ == "__main__":
    #
    # input and model
    #
    input = C.input_variable(shape=(img_channel, img_height, img_width), dtype="float32", needs_gradient=True)
    model = C.load_model("../ActMax/VGG19_ImageNet_Caffe.model")

    vgg19 = C.combine([model.fc8]).clone(method="share", substitutions={model.arguments[0]: input})

    img = cv2.resize(cv2.imread("./cat.jpg"), (img_width, img_height))
    x_img = np.ascontiguousarray(img.transpose(2, 0, 1), dtype="float32") - img_mean

    #
    # vanilla gradients
    #
    vanilla = VanillaGradients(vgg19)

    vanilla_raw = vanilla.saliency_map(x_img)
    vanilla_smooth = vanilla.smooth_grad(x_img)

    #
    # integrated gradients
    #
    integrated = IntegratedGradients(vgg19)

    integrated_raw = integrated.saliency_map(x_img)
    integrated_smooth = integrated.smooth_grad(x_img)

    #
    # occlusion
    #
    occlusion = Occlusion(vgg19)

    occlusion4x4 = grayscale_map(occlusion.saliency_map(x_img, window_size=4))
    occlusion8x8 = grayscale_map(occlusion.saliency_map(x_img, window_size=8))
    occlusion16x16 = grayscale_map(occlusion.saliency_map(x_img, window_size=16))
    occlusion32x32 = grayscale_map(occlusion.saliency_map(x_img, window_size=32))
    
