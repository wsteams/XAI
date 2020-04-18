import cntk as C
import cv2
import matplotlib.pyplot as plt
import numpy as np

from saliency import *

C.debugging.force_deterministic(0)

img_channel = 3
img_height = 224
img_width = 224
img_mean = np.array([[[104]], [[117]], [[124]]], dtype="float32")


def visualize(img_list, title_list, cmap_list):
    plt.figure()
    for i, (img, title, cmap) in enumerate(zip(img_list, title_list, cmap_list)):
        plt.subplot(1, 3, i + 1)
        plt.axis("off")
        plt.imshow(img, cmap=cmap)
        plt.colorbar()
        plt.title(title)
    plt.show()


if __name__ == "__main__":
    #
    # ImageNet categories
    #
    with open("../ImageNet.txt") as f:
        category = f.readlines()

    #
    # input and model
    #
    input = C.input_variable(shape=(img_channel, img_height, img_width), dtype="float32", needs_gradient=True)
    model = C.load_model("../ActMax/VGG19_ImageNet_Caffe.model")

    vgg19 = C.combine([model.fc8]).clone(method="share", substitutions={model.arguments[0]: input})

    img = cv2.resize(cv2.imread("./cat.jpg"), (img_width, img_height))
    x_img = np.ascontiguousarray(img.transpose(2, 0, 1), dtype="float32")

    #
    # classification
    #
    pred = model.eval({model.arguments[0]: x_img - img_mean}).argmax()
    print(category[pred])

    #
    # vanilla gradients
    #
    vanilla = VanillaGradients(vgg19)

    vanilla_raw = vanilla.saliency_map(x_img)
    vanilla_smooth = vanilla.smooth_grad(x_img)

    visualize([img[..., ::-1], grayscale_map(vanilla_raw), grayscale_map(vanilla_smooth)],
              [category[pred][:-1], "Vanilla Gradients", "SmoothGrad"], [None, "hot", "hot"])
    
    #
    # integrated gradients
    #
    integrated = IntegratedGradients(vgg19)

    integrated_raw = integrated.saliency_map(x_img)
    integrated_smooth = integrated.smooth_grad(x_img)

    visualize([img[..., ::-1], grayscale_map(integrated_raw), grayscale_map(integrated_smooth)],
              [category[pred][:-1], "Integrated Gradients", "SmoothGrad"], [None, "hot", "hot"])

    #
    # occlusion
    #
    occlusion = Occlusion(vgg19)

    occlusion4x4 = grayscale_map(occlusion.saliency_map(x_img, window_size=4))
    occlusion8x8 = grayscale_map(occlusion.saliency_map(x_img, window_size=8))
    occlusion16x16 = grayscale_map(occlusion.saliency_map(x_img, window_size=16))
    occlusion32x32 = grayscale_map(occlusion.saliency_map(x_img, window_size=32))

    plt.fig(size=(1800, 634))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.imshow(img[..., ::-1])
    plt.title(category[pred][:-1])
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.imshow(occlusion4x4, cmap="hot")
    plt.colorbar()
    plt.title("Occlusion")
    plt.show()

    visualize([occlusion8x8, occlusion16x16, occlusion32x32],
              ["window size 8x8", "windows size 16x16", "window size 32x32"], ["hot", "hot", "hot"])
    
