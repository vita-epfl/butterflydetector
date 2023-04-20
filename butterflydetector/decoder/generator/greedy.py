from collections import defaultdict
import logging
import time

import numpy as np

from ..annotation import Annotation
from ..utils import scalar_square_add_2dsingle

# pylint: disable=import-error
from ...functional import paf_center, scalar_value, scalar_nonzero, weiszfeld_nd

LOG = logging.getLogger(__name__)


class Greedy(object):
    def __init__(self, pifhr, seeds, *,
                 seed_threshold,
                 debug_visualizer=None):
        self.pifhr = pifhr
        self.seeds = seeds

        self.seed_threshold = seed_threshold

        self.debug_visualizer = debug_visualizer
        self.timers = defaultdict(float)

        if self.debug_visualizer:
            self.debug_visualizer.butterfly_hr(self.pifhr.targets)

    def annotations(self, initial_annotations=None):
        start = time.perf_counter()
        if not initial_annotations:
            initial_annotations = []
        LOG.debug('initial annotations = %d', len(initial_annotations))

        occupied = np.zeros(self.pifhr.scales.shape, dtype=np.uint8)
        annotations = []

        def mark_occupied(ann):
            try:
                for joint_i, xyv in enumerate(ann.data):
                    if xyv[2] == 0.0:
                        continue

                    width = max(4, ann.joint_scales_w[joint_i])
                    height = max(4, ann.joint_scales_h[joint_i])
                    scalar_square_add_2dsingle(occupied[joint_i],
                                             xyv[0],
                                             xyv[1],
                                             width / 2.0, height/2.0, 1)
                                             #np.clip(xyv[3]/5,a_min=2, a_max=None) / 2.0, np.clip(xyv[4]/5,a_min=2, a_max=None)/2, 1)
            except:
                import pdb; pdb.set_trace()

        for ann in initial_annotations:
            if ann.joint_scales is None:
                ann.fill_joint_scales(self.pifhr.scales_w, self.pifhr.scales_h)
            ann.fill_joint_scales(self.pifhr.scales_w, self.pifhr.scales_h)
            annotations.append(ann)
            mark_occupied(ann)

        for v, f, x, y, w, h in self.seeds.get():
            if scalar_nonzero(occupied[f], x, y):
                continue
            ann = Annotation(f, (x, y, v, w, h), self.pifhr.scales_w.shape[0], dim_per_kps=5)
            ann.fill_joint_scales(self.pifhr.scales_w, self.pifhr.scales_h)
            annotations.append(ann)
            mark_occupied(ann)

        if self.debug_visualizer:
            LOG.debug('occupied field 0')
            self.debug_visualizer.occupied(occupied[0])

        LOG.debug('keypoint sets %d, %.3fs', len(annotations), time.perf_counter() - start)
        return annotations
