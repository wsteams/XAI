import cntk as C
import numpy as np


class SaliencyMap:
    """ https://github.com/PAIR-code/saliency/tree/master/saliency """
    def __init__(self, model):
        self.model = model

    def saliency_map(self, img):
        raise NotImplementedError("A super class should implemented get_mask()")

    def smooth_grad(self, img, stdev_spread=0.15, nsamples=25, magnitude=True, **kwargs):
        """ SmoothGrad

        https://arxiv.org/abs/1706.03825

        Parameters
        ----------
        stdev_spread: amount of noise to add to the input, as fraction of the total spread (x_max - x_min) (default=0.15)
        nsamples    : number of samples to average across to get the smooth gradient (default=25)
        magnitude   : If true, computes the sum of squares of gradients instead of just the sum (default=True)

        """
        stdev = stdev_spread * (np.max(img) - np.min(img))

        total_gradients = np.zeros_like(img)
        for i in range(nsamples):
            noise = np.random.normal(0, stdev, img.shape).astype("float32")
            noise_plus = img + noise
            grad = self.saliency_map(noise_plus, **kwargs)[0]
            if magnitude:
                total_gradients += (grad * grad)
            else:
                total_gradients += grad
        return total_gradients / nsamples


class VanillaGradients(SaliencyMap):
    """
    https://arxiv.org/abs/1312.6034
    """
    def __init__(self, model):
        super(VanillaGradients, self).__init__(model)

    def saliency_map(self, img):
        """ Vanilla Gradients """
        pred = int(self.model.eval({self.model.arguments[0]: img})[0].argmax())
        return self.model[pred].grad({self.model.arguments[0]: img})[0]


class IntegratedGradients(VanillaGradients):
    """
    https://arxiv.org/abs/1703.01365
    """
    def saliency_map(self, img, base=None, steps=25):
        """ Integrated Gradients

        Parameters
        ----------
        base  : baseline value used in integration (default=None)
        steps : number of integrated steps between baseline and x (default=25)

        """
        if base is None:
            base = np.zeros_like(img)

        img_diff = img - base

        total_gradients = np.zeros_like(img)
        for step in range(steps):
            img_step = base + (step / steps) * img_diff

            total_gradients += super(IntegratedGradients, self).saliency_map(img_step)[0]

        return img_diff * total_gradients / steps


class Occlusion(SaliencyMap):
    """
    This method slides a window over the image and computes how that occlusion
    affects the class score. When the class score decreases, this is positive
    evidence for the class, otherwise it is negative evidence.
    """
    def __init__(self, model):
        super(Occlusion, self).__init__(model)

    def saliency_map(self, img, window_size=4, fill_value=0):
        """ Occlusion

        Parameters
        ----------
        window_size : occlusion window size (default=4)
        fill_value  : value filled occlusion window (default=0)

        """
        occlusion_window = np.empty((img.shape[0], window_size, window_size))
        occlusion_window.fill(fill_value)

        occlusion_scores = np.zeros_like(img)

        pred = int(self.model.eval({self.model.arguments[0]: img})[0].argmax())
        original_score = self.model[pred].eval({self.model.arguments[0]: img})

        for row in range(img.shape[1] // window_size):
            for col in range(img.shape[2] // window_size):
                occluded = img.copy()
                row_start, row_end = row * window_size, (row + 1) * window_size
                col_start, col_end = col * window_size, (col + 1) * window_size

                occluded[:, row_start:row_end, col_start:col_end] = occlusion_window
                score = self.model[pred].eval({self.model.arguments[0]: occluded})
                score_diff = original_score - score
                occlusion_scores[:, row_start:row_end, col_start:col_end] += score_diff

        return occlusion_scores


def divergence_map(img3d, percentile=99):
    """ Return a gray-scale with positive and negative values """
    img2d = np.sum(img3d, axis=0)

    span = abs(np.percentile(img2d, percentile))
    vmin = -span
    vmax = span

    return np.clip((img2d - vmin) / (vmax - vmin), -1, 1)


def grayscale_map(img3d, percentile=99):
    """ Return a gray-scale image """
    img2d = np.sum(np.abs(img3d), axis=0)

    vmax = np.percentile(img2d, percentile)
    vmin = np.min(img2d)

    return np.clip((img2d - vmin) / (vmax - vmin), 0, 1)

