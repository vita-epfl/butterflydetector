import copy
import logging

import numpy as np
import PIL

from .preprocess import Preprocess
from ..data import BBOX_POINTS, HFLIP_BBOX

LOG = logging.getLogger(__name__)

class HorizontalSwapButterfly():
    def __init__(self, joints=None, hflip=None):
        self.joints = joints or BBOX_POINTS
        self.hflip = hflip or HFLIP_BBOX

    def __call__(self, keypoints):
        target = np.zeros(keypoints.shape)

        for f in range(0, len(keypoints), 5):
            for source_i, xyv in enumerate(keypoints[f:(f+5)]):
                source_name = self.joints[source_i]
                target_name = self.hflip.get(source_name)
                if target_name:
                    target_i = f + self.joints.index(target_name)
                else:
                    target_i = f + source_i
                target[target_i] = xyv

        return target

class HFlip(Preprocess):
    def __init__(self, *, swap=None):
        self.swap = swap or HorizontalSwapButterfly()

    def __call__(self, image, anns, meta):
        meta = copy.deepcopy(meta)
        anns = copy.deepcopy(anns)

        w, _ = image.size
        image = image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        for ann in anns:
            ann['keypoints'][:, 0] = -ann['keypoints'][:, 0] - 1.0 + w
            if 'class' in ann.keys():
                ann['class'][:, 0] = -ann['class'][:, 0] - 1.0 + w
            if self.swap is not None and not ann['iscrowd']:
                ann['keypoints'] = self.swap(ann['keypoints'])
                meta['horizontal_swap'] = self.swap
            ann['bbox'][0] = -(ann['bbox'][0] + ann['bbox'][2]) - 1.0 + w

        assert meta['hflip'] is False
        meta['hflip'] = True

        meta['valid_area'][0] = -(meta['valid_area'][0] + meta['valid_area'][2]) + w
        for ann in anns:
            ann['valid_area'] = meta['valid_area']

        return image, anns, meta
