import logging
import time

import numpy as np

# pylint: disable=import-error
from ..functional import cumulative_average_2d, scalar_square_add_2dgauss_with_max

LOG = logging.getLogger(__name__)


class ButterflyHr(object):
    v_threshold = 0.1

    def __init__(self, scale_wh, pif_nn=None):
        self.pif_nn = pif_nn
        self.scale_wh = scale_wh
        self.target_accumulator = None
        self.scales = None
        self.scales_n = None

        self._clipped = None

    @property
    def targets(self):
        return self.target_accumulator

    def fill(self, pif, stride, min_scale=0.0):
        return self.fill_multiple([pif], stride, min_scale)

    def fill_multiple(self, pifs, stride, min_scale=0.0):
        start = time.perf_counter()

        if self.target_accumulator is None:
            shape = (
                pifs[0].shape[0],
                int((pifs[0].shape[2] - 1) * stride + 1),
                int((pifs[0].shape[3] - 1) * stride + 1),
            )
            ta = np.zeros(shape, dtype=np.float32)
            self.scales = np.zeros(shape, dtype=np.float32)
            self.scales_n = np.zeros(shape, dtype=np.float32)
            self.scales_w = np.zeros(shape, dtype="float32")
            self.scales_h = np.zeros(shape, dtype="float32")
            self.widths = np.zeros(shape, dtype="float32")
            self.heights = np.zeros(shape, dtype="float32")
            self.scalew_n = np.zeros(shape, dtype="float32")
            self.scaleh_n = np.zeros(shape, dtype="float32")
            self.width_n = np.zeros(shape, dtype="float32")
            self.height_n = np.zeros(shape, dtype="float32")
        else:
            ta = np.zeros(self.target_accumulator.shape, dtype=np.float32)
        scale_div = {(k+1):5 for k in range(shape[0])}
        for pif in pifs:
            for field_numb, (t, p, scale_w, scale_h, width, height, n_sw, n_sh, n_w, n_h) in enumerate(zip(ta, pif, self.scales_w, self.scales_h, self.widths, self.heights, self.scalew_n, self.scaleh_n, self.width_n, self.height_n)):
                p = p[:, p[0] > self.v_threshold]
                if min_scale:
                    p = p[:, p[3] > min_scale / stride]

                v, x, y, w, h = p
                x = x * stride
                y = y * stride
                w = np.exp(w)
                h = np.exp(h)
                w = w * self.scale_wh
                h = h * self.scale_wh
                if np.isinf(w).any():
                    mask = np.logical_not(np.isinf(w))
                    x = x[mask]
                    y = y[mask]
                    v = v[mask]
                    h = h[mask]
                    w = w[mask]

                if np.isinf(h).any():
                    mask = np.logical_not(np.isinf(h))
                    w = w[mask]
                    x = x[mask]
                    y = y[mask]
                    v = v[mask]
                    h = h[mask]
                s_h = np.clip(h/scale_div[field_numb+1], a_min=2, a_max=None)
                s_w = np.clip(w/scale_div[field_numb+1], a_min=2, a_max=None)
                if self.pif_nn:
                    pif_nn = self.pif_nn
                else:
                    #pif_nn = np.clip((w/self.scale_wh)*(h/self.scale_wh), a_min=1, a_max= None)
                    pif_nn_w = np.copy(np.clip(w/self.scale_wh, a_min=1, a_max= None))#.astype(np.int)
                    pif_nn_h = np.copy(np.clip(h/self.scale_wh, a_min=1, a_max= None))#.astype(np.int)
                    #pif_nn_w[pif_nn_w>16] = pif_nn_w[pif_nn_w>16]*0.2
                    #pif_nn_h[pif_nn_h>16] = pif_nn_h[pif_nn_h>16]*0.2
                    pif_nn = np.clip(pif_nn_w*pif_nn_h, a_min=1, a_max= None)
                scalar_square_add_2dgauss_with_max(t, x, y, s_w, s_h, (v / pif_nn / len(pifs)).astype(np.float32), truncate=0.5, max_value=100000.0)

                cumulative_average_2d(scale_w, n_sw, x, y, s_w, s_h, (s_w), v)
                cumulative_average_2d(scale_h, n_sh, x, y, s_w, s_h, (s_h), v)
                cumulative_average_2d(width, n_w, x, y, s_w, s_h, w, v)
                cumulative_average_2d(height, n_h, x, y, s_w, s_h, h, v)
                # scalar_square_add_gauss_with_max(
                #     t, x, y, s, v / pif_nn / len(pifs), truncate=1.0)
                # cumulative_average(scale, n, x, y, s, s, v)
        ta = np.tanh(ta)
        if self.target_accumulator is None:
            self.target_accumulator = ta
        else:
            self.target_accumulator = np.maximum(ta, self.target_accumulator)

        LOG.debug('target_intensities %.3fs', time.perf_counter() - start)
        return self

    def fill_sequence(self, pifs, strides, min_scales):
        if len(pifs) == 10:
            for pif1, pif2, stride, min_scale in zip(pifs[:5], pifs[5:], strides, min_scales):
                self.fill_multiple([pif1, pif2], stride, min_scale=min_scale)
        else:
            for pif, stride, min_scale in zip(pifs, strides, min_scales):
                self.fill(pif, stride, min_scale=min_scale)

        return self
