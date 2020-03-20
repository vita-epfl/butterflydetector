"""Decoder for pif-paf fields."""

import logging
import time

from . import generator
from .butterfly_hr import ButterflyHr
from .butterfly_seeds import ButterflySeeds
from .utils import normalize_butterfly

LOG = logging.getLogger(__name__)


class Butterfly(object):
    pif_fixed_scale = None
    scale_div = 10

    def __init__(self, stride, *,
                 pif_index=0,
                 head_names=None,
                 pif_min_scale=0.0,
                 seed_threshold=0.2,
                 debug_visualizer=None):
        self.strides = stride
        self.scale_wh = stride
        self.pif_min_scales = pif_min_scale
        if 'nsbutterfly' in head_names:
            self.scale_wh = 1
        self.pif_indices = pif_index
        if not isinstance(self.strides, (list, tuple)):
            self.strides = [self.strides]
            self.pif_indices = [self.pif_indices]
        if not isinstance(self.pif_min_scales, (list, tuple)):
            self.pif_min_scales = [self.pif_min_scales for _ in self.strides]
        assert len(self.strides) == len(self.pif_indices)
        assert len(self.strides) == len(self.pif_min_scales)


        self.seed_threshold = seed_threshold
        self.debug_visualizer = debug_visualizer

        self.pif_nn = 16
        if 'obutterfly' in head_names:
            self.pif_nn_thres = 1

    def __call__(self, fields, initial_annotations=None):
        start = time.perf_counter()

        if self.debug_visualizer:
            for stride, pif_i in zip(self.strides, self.pif_indices):
                self.debug_visualizer.butterfly_raw(fields[pif_i], stride)
        # normalize
        normalized_pifs = [normalize_butterfly(*fields[pif_i], fixed_scale=self.pif_fixed_scale)[0]
                           for pif_i in self.pif_indices]

        # pif hr
        pifhr = ButterflyHr(self.scale_wh, self.pif_nn)
        pifhr.fill_sequence(normalized_pifs, self.strides, self.pif_min_scales)

        # seeds
        seeds = ButterflySeeds(pifhr, self.seed_threshold,
                         debug_visualizer=self.debug_visualizer)
        seeds.fill_sequence(normalized_pifs, self.strides, self.pif_min_scales)

        # paf_scored

        gen = generator.Greedy(
            pifhr, seeds,
            seed_threshold=self.seed_threshold,
            debug_visualizer=self.debug_visualizer,
        )

        annotations = gen.annotations(initial_annotations=initial_annotations)
        # if self.force_complete:
        #     annotations = gen.complete_annotations(annotations)

        LOG.debug('annotations %d, %.3fs', len(annotations), time.perf_counter() - start)
        return annotations
