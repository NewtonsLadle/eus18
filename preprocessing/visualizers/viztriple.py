import scipy.misc as misc
from pathlib import Path
import numpy as np

truth_color = [142, 0, 0]
prediction_color = [226, 164, 20]

class VizTriple(object):

    def __init__(self, floor, ceiling):
        self.floor = floor
        self.ceiling = ceiling


    def create_triple(self, location, volume, annotation, truth, prediction):
        volume[volume < self.floor] = self.floor
        volume[volume > self.ceiling] = self.ceiling
        volume = ((volume - self.floor)/(self.ceiling - self.floor)*255).astype(np.uint8)

        base = np.stack(
            (volume, volume, volume),
            axis=2
        )
        annotation_image = base.copy()
        annotation_image[annotation[:,:,0]!=0,:] = truth_color
        print(np.shape(annotation_image))
        misc.imsave(str(location / "annotation.png"), annotation_image)

        truth_image = base.copy()
        truth_image[truth[:,:]!=0,:] = truth_color
        misc.imsave(str(location / "truth.png"), truth_image)

        pred_image = base.copy()
        pred_image[prediction[:,:]!=0,:] = prediction_color
        misc.imsave(str(location / "prediction.png"), pred_image)
