"""Decoder for pif fields."""

from collections import defaultdict
import logging
import time

import numpy as np

from .annotation import Annotation
from .utils import index_field, scalar_square_add_2dsingle, normalize_butterfly

# pylint: disable=import-error
from ..functional import (scalar_square_add_2dgauss, scalar_square_add_2dconstant, cumulative_average_2d)

import re
class Butterfly(object):
    default_pif_fixed_scale = None
    scale_div = 10
    def __init__(self, stride, seed_threshold,
                 head_index=None,
                 head_names=None,
                 profile=None,
                 debug_visualizer=None,
                 **kwargs):
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.debug('unused arguments %s', kwargs)

        self.stride = stride
        self.scale_wh = self.stride
        if 'nsbutterfly' in head_names:
            self.scale_wh = 1
        self.head_index = head_index or 0
        self.profile = profile
        self.seed_threshold = seed_threshold
        self.debug_visualizer = debug_visualizer
        self.pif_fixed_scale = self.default_pif_fixed_scale

        self.pif_nn_thres = 16
        if 'obutterfly' in head_names:
            self.pif_nn_thres = 1

    def __call__(self, fields, initial_annotations=None):
        start = time.perf_counter()
        if self.profile is not None:
            self.profile.enable()

        pif = fields[self.head_index]
        if self.debug_visualizer:
            self.debug_visualizer.butterfly_raw(pif, self.stride)
        pif, pif_spread = normalize_butterfly(*pif, fixed_scale=self.pif_fixed_scale)
        if True:
            ButterflyGenerator.scale_div = self.scale_div
            gen = ButterflyGenerator(
                pif,
                pif_spread= pif_spread,
                stride=self.stride,
                scale_wh = self.scale_wh,
                seed_threshold=self.seed_threshold,
                pif_nn_thres=self.pif_nn_thres,
                debug_visualizer=self.debug_visualizer,
            )

            annotations = gen.annotations()
        else:
            #seeds = np.empty((0, 6))
            seeds_list = []
            scale_w = []
            scale_h = []
            for field_i, p in enumerate(pif):
                v, x, y, w, h = p[:, p[0] > self.seed_threshold]
                w = np.exp(w) #/ scale_wh
                h = np.exp(h) #/ scale_wh
                x = x * self.stride
                y = y * self.stride
                w = w * self.scale_wh
                h = h * self.scale_wh
                s_h = np.clip(h/10, a_min=1, a_max=10)
                s_w = np.clip(w/10, a_min=1, a_max=10)


                f = np.full((v.shape[0],), field_i)
                candidate = np.column_stack((v, f, x, y, w, h))
                #seeds = np.concatenate((seeds, candidate))
                seeds_list.append(candidate)
                scale_w.append(s_w)
                scale_h.append(s_h)

            scale_w = np.asarray(scale_w)
            scale_h = np.asarray(scale_h)
            annotations = []
            for seeds in seeds_list:
                for index, (v, f, x, y, w, h) in enumerate(seeds):
                    ann = Annotation(int(f), (x, y, v, w, h), pif.shape[0], dim_per_kps=5)
                    ann.fill_joint_scales_nothr(scale_w, scale_h, index)
                    annotations.append(ann)

        self.log.debug('annotations %d, %.3fs', len(annotations), time.perf_counter() - start)
        if self.profile is not None:
            self.profile.disable()
        return annotations


class ButterflyGenerator(object):
    scale_div = 5
    def __init__(self, pif_field, *,
                 pif_spread,
                 stride,
                 scale_wh,
                 seed_threshold,
                 pif_nn_thres,
                 debug_visualizer=None):
        self.log = logging.getLogger(self.__class__.__name__)

        self.pif = pif_field
        self.pif_spread = pif_spread
        self.stride = stride
        self.scale_wh = scale_wh
        self.seed_threshold = seed_threshold
        self.pif_nn_thres = pif_nn_thres
        self.debug_visualizer = debug_visualizer
        self.timers = defaultdict(float)

        # pif init
        self._pifhr, self._pifhr_scales_w, self._pifhr_scales_h, self._pifhr_width, self._pifhr_height = self._target_intensities()
        if self.debug_visualizer:
            self.debug_visualizer.butterfly_hr(self._pifhr)
            self.debug_visualizer.butterflyhr_wh(self._pifhr_width, self._pifhr_height, self._pifhr)

    def _target_intensities(self, v_th=0.1):
        start = time.perf_counter()
        xy_offset = [-0.5, -0.5]
        targets = np.zeros((self.pif.shape[0],
                            int(self.pif.shape[2] * self.stride),
                            int(self.pif.shape[3] * self.stride)), dtype="float32")
        scales_w = np.zeros(targets.shape, dtype="float32")
        scales_h = np.zeros(targets.shape, dtype="float32")
        widths = np.zeros(targets.shape, dtype="float32")
        heights = np.zeros(targets.shape, dtype="float32")
        scalew_n = np.zeros(targets.shape, dtype="float32")
        scaleh_n = np.zeros(targets.shape, dtype="float32")
        width_n = np.zeros(targets.shape, dtype="float32")
        height_n = np.zeros(targets.shape, dtype="float32")
        #ns_average = np.zeros(targets.shape, dtype="float32")
        #scale_div = {1: 10, 2: 5, 3: 5, 4: 5, 5: 5, 6: 5, 7: 5, 8: 1, 9: 15, 10: 5}

        #scale_div = {1: 5, 2: 5, 3: 5, 4: 5, 5: 5, 6: 5, 7: 5, 8: 5, 9: 5, 10: 5}
        scale_div = {(k+1):5 for k in range(self.pif.shape[0])}
        for field_numb, (t, p, scale_w, scale_h, width, height, n_sw, n_sh, n_w, n_h) in enumerate(zip(targets, self.pif, scales_w, scales_h, widths, heights, scalew_n, scaleh_n, width_n, height_n)):
            v, x, y, w, h = p[:, p[0] > v_th]
            w = np.exp(w) #/ scale_wh
            h = np.exp(h) #/ scale_wh

            # minx, miny, maxx, maxy = np.round(x - w/2-0.5).astype(np.int), np.round(y - h/2-0.5).astype(np.int), np.round(x + w/2+0-5).astype(np.int), np.round(y + h/2+0.5).astype(np.int)
            # piff_nn_w = maxx - minx
            # piff_nn_h = maxy - miny
            # piff_nn_w += piff_nn_w%2
            # piff_nn_h += piff_nn_h%2
            # pif_nn = piff_nn_w * piff_nn_h
            # pif_nn[pif_nn== 0] = (targets.shape[1]*targets.shape[2])
            # pif_nn = pif_nn.astype(np.int32)
            #pif_nn = (w/self.stride)*(h/self.stride)
            #pif_nn = (w*(self.scale_wh/self.stride))*(h*(self.scale_wh/self.stride))
            #import pdb; pdb.set_trace()
            #pif_nn[pif_nn<self.pif_nn_thres] = self.pif_nn_thres
            #pif_nn = np.ones(w.shape)
            x = x * self.stride
            y = y * self.stride
            #s = np.sqrt(np.multiply(w*(self.scale_wh/self.stride), h*(self.scale_wh/self.stride))) * self.stride#/10

            #s_w =  np.clip(w*0.5, a_min=4.0, a_max=None)
            #s_h =  np.clip(h*0.5, a_min=4.0, a_max=None)
            #import pdb; pdb.set_trace()
            #s[s>1] = np.log(10*s[s>1])
            #s = np.sqrt(s)
            #s = np.clip(s, a_min=4.0, a_max=None)
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
            #s_w = w/10 #np.clip(w/4, a_min=None, a_max=None)
            #s_h = h/10 #np.clip(h/4, a_min=None, a_max=None)
            #s_w = 4/(1+np.exp(-0.2*(w-10)))
            #s_h = 4/(1+np.exp(-0.2*(h-10)))

            #s_h = np.clip(h/self.scale_div, a_min=2, a_max=None)
            #s_w = np.clip(w/self.scale_div, a_min=2, a_max=None)

            s_h = np.clip(h/scale_div[field_numb+1], a_min=2, a_max=None)
            s_w = np.clip(w/scale_div[field_numb+1], a_min=2, a_max=None)



            #s_h = np.copy(h)
            #s_w = np.copy(w)
            clip = 60
            multiplier = 4
            #s_w[s_w<clip] = s_w[s_w<clip]/10
            #s_w[s_w>=clip] = np.sqrt(multiplier*s_w[s_w>=clip]) - 11
            #s_h[s_h<clip] = s_h[s_h<clip]/10
            #s_h[s_h>=clip] = np.sqrt(multiplier*s_h[s_h>=clip]) - 11
            #s_w = w * (0.5 - w/targets.shape[2])/2
            #s_h = h * (0.5 - h/targets.shape[1])/2

            cumulative_average_2d(width, n_w, x, y, s_w, s_h, w, v)
            cumulative_average_2d(height, n_h, x, y, s_w, s_h, h, v)
            #index_x = np.clip(x.astype(np.long), a_min=0, a_max= height.shape[1]-1)
            #index_y = np.clip(y.astype(np.long), a_min=0, a_max= height.shape[0]-1)
            # #
            #pif_nn_w = (width[index_y, index_x]/self).scale_wh.astype(np.int)
            #pif_nn_h = (height[index_y, index_x]/self.scale_wh).astype(np.int)
            # # pif_nn_w[pif_nn_w>16] = pif_nn_w[pif_nn_w>16]*0.2
            # # pif_nn_h[pif_nn_h>16] = pif_nn_h[pif_nn_h>16]*0.2
            #pif_nn = np.clip((w/self.scale_wh)*(h/self.scale_wh), a_min=1, a_max= None)
            #pif_nn = np.clip(pif_nn_w*pif_nn_h, a_min=1, a_max= None)
            pif_nn=16
            scalar_square_add_2dgauss(t, x, y, s_w, s_h, (v / pif_nn).astype(np.float32), truncate=0.5)

            # scalar_square_add_2dconstant(scale_w, x, y, s_w, s_h, (s_w)*v)
            # scalar_square_add_2dconstant(scale_h, x, y, s_w, s_h, (s_h)*v)
            # scalar_square_add_2dconstant(width, x, y, s_w, s_h, w*v)
            # scalar_square_add_2dconstant(height, x, y, s_w, s_h, h*v)

            cumulative_average_2d(scale_w, n_sw, x, y, s_w, s_h, (s_w), v)
            cumulative_average_2d(scale_h, n_sh, x, y, s_w, s_h, (s_h), v)

            #scalar_square_add_2dconstant(width, x, y, s_w, s_h, w)
            #scalar_square_add_2dconstant(height, x, y, s_w, s_h, h)

            #scalar_square_add_2dconstant(n, x, y, s_w, s_h, v)
            #scalar_square_add_2dconstant(n_average, x, y, s_w, s_h, np.ones(w.shape, dtype="float32"))

            # cumulative_average_2d(scale_w, n, x, y, s_w, s_h, (s_w/2), v)
            # cumulative_average_2d(scale_h, n, x, y, s_w, s_h, (s_h/2), v)
            # cumulative_average_2d(width, n, x, y, s_w, s_h, w, v)
            # cumulative_average_2d(height, n, x, y, s_w, s_h, h, v)

        #targets = np.minimum(1.0, targets)
        #targets = 1/(1 + np.exp(-targets))
        targets = np.tanh(targets)

        #m = ns > 0
        #widths[m] = widths[m] / ns[m]
        #heights[m] = heights[m] / ns[m]

        #widths[m] = widths[m] / ns_average[m]
        #heights[m] = heights[m] / ns_average[m]

        #scales_w[m] = scales_w[m] / ns[m]
        #scales_h[m] = scales_h[m] / ns[m]

        self.log.debug('target_intensities %.3fs', time.perf_counter() - start)
        return targets, scales_w, scales_h, widths, heights

    def annotations(self):
        start = time.perf_counter()

        seeds = self._pifhr_seeds()
        annotations = []
        for v, f, x, y, w, h in seeds:
            ann = Annotationbutterfly(f, (x, y, v, w, h), self._pifhr_scales_w.shape[0], dim_per_kps=5)
            ann.fill_joint_scales(self._pifhr_scales_w, self._pifhr_scales_h)
            #ann.data[:, 0:2] *= self.stride
            #ann.data[:, 3:5] *= self.stride
            annotations.append(ann)
        self.log.debug('keypoint sets %d, %.3fs', len(annotations), time.perf_counter() - start)
        return annotations

    def _pifhr_seeds(self):
        start = time.perf_counter()
        seeds = []
        try:
            for field_i, (f, s_w, s_h, w, h) in enumerate(zip(self._pifhr, self._pifhr_scales_w, self._pifhr_scales_h, self._pifhr_width, self._pifhr_height)):
                index_fields = index_field(f.shape)
                candidates = np.concatenate((index_fields, np.expand_dims(f, 0), np.expand_dims(w, 0), np.expand_dims(h, 0)), 0)

                mask = f > self.seed_threshold
                candidates = np.moveaxis(candidates[:, mask], 0, -1)

                occupied = np.zeros(s_w.shape)
                for c in sorted(candidates, key=lambda c: c[2], reverse=True):
                    i, j = int(c[0]), int(c[1])
                    if occupied[j, i]:
                        continue

                    width = max(4, s_w[j, i])
                    height = max(4, s_h[j, i])
                    scalar_square_add_2dsingle(occupied, c[0], c[1], width / 2.0, height/2, 1.0)
                    #scalar_square_add_2dsingle(occupied, c[0], c[1], np.clip(c[3]/5,a_min=2, a_max=None) / 2.0, np.clip(c[4]/5,a_min=2, a_max=None)/2, 1)
                    seeds.append((c[2], field_i, c[0] , c[1], c[3], c[4]))

                if self.debug_visualizer:
                    if field_i in self.debug_visualizer.pif_indices:
                        self.log.debug('occupied seed, field %d', field_i)
                        self.debug_visualizer.occupied(occupied)
        except:
            import pdb; pdb.set_trace()

        seeds = list(sorted(seeds, reverse=True))
        if len(seeds) > 500:
            if seeds[500][0] > 0.1:
                seeds = [s for s in seeds if s[0] > 0.1]
            else:
                seeds = seeds[:500]
        if self.debug_visualizer:
            self.debug_visualizer.seeds_butterfly(seeds, self.stride)

        self.log.debug('seeds %d, %.3fs', len(seeds), time.perf_counter() - start)
        return seeds
