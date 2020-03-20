import logging
import time

# pylint: disable=import-error
from ..functional import scalar_values
import numpy as np

LOG = logging.getLogger(__name__)


class ButterflySeeds(object):
    def __init__(self, pifhr, seed_threshold, *,
                 debug_visualizer=None):
        self.pifhr = pifhr
        self.seed_threshold = seed_threshold
        self.debug_visualizer = debug_visualizer

        self.seeds = []

    def fill(self, pif, stride, min_scale=0.0):
        self.stride = stride
        start = time.perf_counter()

        for field_i, p in enumerate(pif):
            #p = p[:, p[0] > self.seed_threshold / 2.0]
            p = p[:, p[0] > min(0.1, self.seed_threshold / 2.0)]
            #p = p[:, p[0] >0]
            if min_scale:
                p = p[:, p[3] > min_scale / stride]
            v, x, y, w, h = p
            v = scalar_values(self.pifhr.target_accumulator[field_i], x * stride, y * stride)
            w = scalar_values(self.pifhr.widths[field_i], x * stride, y * stride)
            h = scalar_values(self.pifhr.heights[field_i], x * stride, y * stride)
            m = v > self.seed_threshold
            x, y, v, w, h = x[m] * stride, y[m] * stride, v[m], w[m], h[m]#np.exp(w[m]) * stride, np.exp(h[m])*stride #w[m], h[m]

            for vv, xx, yy, ww, hh in zip(v, x, y, w, h):
                self.seeds.append((vv, field_i, xx, yy, ww, hh))
        LOG.debug('seeds %d, %.3fs', len(self.seeds), time.perf_counter() - start)
        return self

    def get(self):
        if self.debug_visualizer:
            self.debug_visualizer.seeds_butterfly(self.seeds, self.stride)

        seeds = sorted(self.seeds, reverse=True)
        if len(seeds) > 500:
            if seeds[500][0] > 0.1:
                seeds = [s for s in seeds if s[0] > 0.1]
            else:
                seeds = seeds[:500]
        return seeds

    def fill_sequence(self, pifs, strides, min_scales):
        for pif, stride, min_scale in zip(pifs, strides, min_scales):
            self.fill(pif, stride, min_scale=min_scale)

        return self
