from abc import ABCMeta, abstractmethod
import copy
import numpy as np

class Preprocess(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, image, anns, meta):
        """Implementation of preprocess operation."""

    @staticmethod
    def keypoint_sets_inverse(keypoint_sets, meta):
        keypoint_sets = keypoint_sets.copy()

        keypoint_sets[:, :, 0] += meta['offset'][0]
        keypoint_sets[:, :, 1] += meta['offset'][1]

        keypoint_sets[:, :, 0] = keypoint_sets[:, :, 0] / meta['scale'][0]
        keypoint_sets[:, :, 1] = keypoint_sets[:, :, 1] / meta['scale'][1]

        if meta['hflip']:
            w = meta['width_height'][0]
            keypoint_sets[:, :, 0] = -keypoint_sets[:, :, 0] + (w - 1)
            for keypoints in keypoint_sets:
                if meta.get('horizontal_swap'):
                    keypoints[:] = meta['horizontal_swap'](keypoints)

        return keypoint_sets

    @staticmethod
    def bbox_sets_inverse(bboxes, meta):
        bboxes = bboxes.copy()

        bboxes[:, :, 0] += meta['offset'][0]
        bboxes[:, :, 1] += meta['offset'][1]

        bboxes[:, :, 0] = bboxes[:, :, 0] / meta['scale'][0]
        bboxes[:, :, 1] = bboxes[:, :, 1] / meta['scale'][1]
        bboxes[:, :, 2] = bboxes[:, :, 2] / meta['scale'][0]
        bboxes[:, :, 3] = bboxes[:, :, 3] / meta['scale'][1]

        if meta['hflip']:
            w = meta['width_height'][0]
            bboxes[:, :, 0] = -bboxes[:, :, 0] + (w - 1)
            for bbox in bboxes:
                if meta.get('horizontal_swap'):
                    bbox[:] = meta['horizontal_swap'](bbox)

        return bboxes

    @staticmethod
    def annotations_inverse(annotations, meta):
        annotations = copy.deepcopy(annotations)

        for ann in annotations:
            ann.data[:, 0] += meta['offset'][0]
            ann.data[:, 1] += meta['offset'][1]

            ann.data[:, 0] = ann.data[:, 0] / meta['scale'][0]
            ann.data[:, 1] = ann.data[:, 1] / meta['scale'][1]

            if meta['hflip']:
                w = meta['width_height'][0]
                ann.data[:, 0] = -ann.data[:, 0] + (w - 1)
                if meta.get('horizontal_swap'):
                    ann.data[:] = meta['horizontal_swap'](ann.data)

        for ann in annotations:
            for _, __, c1, c2 in ann.decoding_order:
                c1[:2] += meta['offset']
                c2[:2] += meta['offset']

                c1[:2] /= meta['scale']
                c2[:2] /= meta['scale']

        keypoint_sets = [ann.data[:, :3] for ann in annotations]
        bboxes = [np.array([ann.data[:, 0]-ann.data[:, 3]/2, ann.data[:, 1]-ann.data[:, 4]/2, ann.data[:, 3], ann.data[:, 4]]).transpose() for ann in annotations]
        scores = [ann.score() for ann in annotations]
        if not keypoint_sets:
            return annotations, np.zeros((0, 17, 3)), np.zeros((0, 1,4)), np.zeros((0,))
        keypoint_sets = np.array(keypoint_sets)
        scores = np.array(scores)
        bboxes = np.array(bboxes)

        return annotations, keypoint_sets, bboxes, scores
