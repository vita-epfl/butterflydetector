import logging

import numpy as np
import scipy.ndimage
import torch

from .annrescaler import AnnRescaler
from ..utils import create_sink, mask_valid_area, create_sink_2d


class Butterfly(object):
    side_length = 1

    def __init__(self, stride, scale_wh,
                 n_keypoints=None,
                 v_threshold=0,
                 obutterfly=False,
                 **kwargs):
        self.log = logging.getLogger(self.__class__.__name__)

        self.stride = stride
        self.obutterfly = obutterfly
        self.scale_wh = scale_wh
        self.n_keypoints = n_keypoints
        self.v_threshold = v_threshold

    def __call__(self, anns, width_height_original):
        rescaler = AnnRescaler(self.stride, self.n_keypoints)
        keypoint_sets, bg_mask, valid_area = rescaler(anns, width_height_original)
        self.log.debug('valid area: %s, butterfly side length = %d', valid_area, self.side_length)

        n_fields = self.n_keypoints
        f = ButterflyGenerator(self.side_length, self.stride, self.scale_wh, self.v_threshold, obutterfly=self.obutterfly)
        f.init_fields(n_fields, bg_mask)
        f.fill(keypoint_sets)
        return f.fields(valid_area)


class ButterflyGenerator(object):
    def __init__(self, side_length, stride, scale_wh, v_threshold, obutterfly=False, padding=2):
        self.side_length = side_length
        self.v_threshold = v_threshold
        self.obutterfly = obutterfly
        self.padding = padding
        self.stride = stride
        self.scale_wh = scale_wh
        self.intensities = None
        self.fields_reg = None
        self.fields_scale = None
        self.fields_reg_l = None

        self.s_offset = None
        if self.side_length > -1:
            self.sink = create_sink(side_length)
            self.s_offset = (self.side_length - 1.0) / 2.0
            if self.s_offset == 0:
                self.s_offset = (self.side_length- 1.0) / 2.0


        self.log = logging.getLogger(self.__class__.__name__)

    def init_fields(self, n_fields, bg_mask):
        field_w = bg_mask.shape[1] + 2 * self.padding
        field_h = bg_mask.shape[0] + 2 * self.padding
        self.intensities = np.zeros((n_fields + 1, field_h, field_w), dtype=np.float32)
        self.fields_reg = np.zeros((n_fields, 2, field_h, field_w), dtype=np.float32)
        self.fields_width = np.full((n_fields, field_h, field_w), np.nan, dtype=np.float32)
        #self.fields_scale = np.zeros((n_fields, field_h, field_w), dtype=np.float32)
        self.fields_height = np.full((n_fields, field_h, field_w), np.nan, dtype=np.float32)
        self.fields_reg_l = np.full((n_fields, field_h, field_w), np.inf, dtype=np.float32)

        # bg_mask
        self.intensities[-1] = 1.0
        self.intensities[-1, self.padding:-self.padding, self.padding:-self.padding] = bg_mask
        # self.intensities[-1] = scipy.ndimage.binary_erosion(self.intensities[-1],
        #                                                     iterations=1 + 1,
        #                                                     border_value=1)

    def fill(self, keypoint_sets):
        for keypoints in keypoint_sets:
            self.fill_keypoints(keypoints)

    def fill_keypoints(self, keypoints):
        visible = keypoints[:, 2] > 0
        f = np.argmax(np.sum(np.reshape(visible, (-1, 5)), axis=1))
        if not np.any(visible):
            return

        width = (np.max(keypoints[visible, 0]*self.scale_wh) - np.min(keypoints[visible, 0]*self.scale_wh))
        height = (np.max(keypoints[visible, 1]*self.scale_wh) - np.min(keypoints[visible, 1]*self.scale_wh))
        area = (
            (np.max(keypoints[visible, 0]) - np.min(keypoints[visible, 0])) *
            (np.max(keypoints[visible, 1]) - np.min(keypoints[visible, 1]))
        )
        scale = np.sqrt(area)/10
        self.log.debug('instance scale = %.3f', scale)

        xyv = keypoints[visible][-1]
        #f = len(keypoints)-1
        if xyv[2] <= self.v_threshold:
            return

        if self.side_length == -1:
            self.fill_coordinate_kps2(f, keypoints[visible], width, height, scale)
        elif self.side_length == -2:
            self.fill_coordinate_max4(f, keypoints[visible], width, height, scale)
        elif self.side_length == -3:
            self.fill_coordinate_kpsGradient(f, keypoints[visible], width, height, scale)
        else:
            self.fill_coordinate(f, xyv, width, height, scale)

    def fill_coordinate(self, f, xyv, width, height, scale):
        '''
        Use a normal 4x4 field
        '''
        #import pdb; pdb.set_trace()
        ij = np.round(xyv[:2] - self.s_offset).astype(np.int) + self.padding
        minx, miny = int(ij[0]), int(ij[1])
        maxx, maxy = minx + self.side_length, miny + self.side_length
        if minx < 0 or maxx > self.intensities.shape[2] or \
           miny < 0 or maxy > self.intensities.shape[1]:
            return
        offset = xyv[:2] - (ij + self.s_offset - self.padding)
        offset = offset.reshape(2, 1, 1)

        # update intensity
        self.intensities[f, miny:maxy, minx:maxx] = 1.0
        # update regression
        sink_reg = self.sink + offset
        sink_l = np.linalg.norm(sink_reg, axis=0)
        mask = sink_l < self.fields_reg_l[f, miny:maxy, minx:maxx]
        self.fields_reg[f, :, miny:maxy, minx:maxx][:, mask] = \
            sink_reg[:, mask]
        self.fields_reg_l[f, miny:maxy, minx:maxx][mask] = sink_l[mask]

        # update width and height
        self.fields_width[f, miny:maxy, minx:maxx][mask] = np.log(width)
        self.fields_height[f, miny:maxy, minx:maxx][mask] = np.log(height)

    def fill_coordinate_kps2(self, f, kps, width, height, scale):
        '''
        Use a wxh field pointing towards the center
        '''
        xy_offset = [-0.5, -0.5]
        #xy_offset = [0, 0]
        minx, miny = np.min(kps[kps[:,2]>0, 0]), np.min(kps[kps[:,2]>0, 1])
        maxx, maxy = np.max(kps[kps[:,2]>0, 0]), np.max(kps[kps[:,2]>0, 1])
        w = np.round(maxx - minx + 0.5).astype(np.int)

        h = np.round(maxy- miny + 0.5).astype(np.int)

        xyv = kps[-1]
        xy_offset = [(w - 1.0) / 2.0, (h - 1.0) / 2.0]

        ij = np.round(xyv[:2] - xy_offset).astype(np.int) + self.padding
        offset = xyv[:2] - (ij + xy_offset - self.padding)
        minx, miny = int(ij[0]), int(ij[1])

        if not self.obutterfly:
            if w<=0:
                raise Exception('w= ', w, ' is negative or zero')
            if h<=0:
                raise Exception('h= ', h, ' is negative or zero')
        else:
            if w==0:
                w = 1
            if h==0:
                h=1

        maxx, maxy = minx + w, miny + h

        sink = create_sink_2d(w, h)

        if minx + w/2 < 0 or maxx - w/2 > self.intensities.shape[2] or \
           miny + h/2 < 0 or maxy - h/2 > self.intensities.shape[1]:
            return
        if False:
            if w > 16 or h>16:
                sigma_ignore, sigma_eff = 0.5, 0.2
                w_ignore, h_ignore = np.round(sigma_ignore*w + 0.5).astype(np.int), np.round(sigma_ignore*h+ 0.5).astype(np.int)
                xy_offset = [(w_ignore - 1.0) / 2.0, (h_ignore - 1.0) / 2.0]
                ij = np.round(xyv[:2] - xy_offset).astype(np.int) + self.padding
                if w>16:
                    minx_ignore = int(ij[0])
                else:
                    minx_ignore = minx
                    w_ignore = w
                if h>16:
                    miny_ignore = int(ij[1])
                else:
                    miny_ignore = miny
                    h_ignore = h
                maxx_ignore, maxy_ignore = minx_ignore + w_ignore, miny_ignore + h_ignore
                self.intensities[f, miny_ignore:maxy_ignore, minx_ignore:maxx_ignore] = np.nan

                w_ignore, h_ignore = np.round(sigma_eff*w + 0.5).astype(np.int), np.round(sigma_eff*h+ 0.5).astype(np.int)
                xy_offset = [(w_ignore - 1.0) / 2.0, (h_ignore - 1.0) / 2.0]
                ij = np.round(xyv[:2] - xy_offset).astype(np.int) + self.padding
                if w>16:
                    minx = int(ij[0])
                else:
                    w_ignore = w
                if h>16:
                    miny = int(ij[1])
                else:
                    h_ignore = h
                ij = np.array([minx, miny], dtype=np.float32)
                maxx, maxy = minx + w_ignore, miny + h_ignore
                xy_offset = [(w_ignore - 1.0) / 2.0, (h_ignore - 1.0) / 2.0]
                offset = xyv[:2] - (ij + xy_offset - self.padding)
                sink = create_sink_2d(w_ignore, h_ignore)

        minx_n = max(0, minx)
        miny_n = max(0, miny)
        maxx_n = min(maxx, self.intensities.shape[2])
        maxy_n = min(maxy, self.intensities.shape[1])
        sink = sink[:, (miny_n-miny):(miny_n-miny) + (maxy_n-miny_n), (minx_n-minx):(minx_n-minx) + (maxx_n-minx_n)]
        minx = minx_n
        maxx = maxx_n
        miny = miny_n
        maxy = maxy_n
        offset = offset.reshape(2, 1, 1)

        # update intensity
        self.intensities[f, miny:maxy, minx:maxx] = 1.0
        # update regression
        sink_reg = sink + offset
        sink_l = np.linalg.norm(sink_reg, axis=0)
        mask = sink_l < self.fields_reg_l[f, miny:maxy, minx:maxx]
        self.fields_reg[f, :, miny:maxy, minx:maxx][:, mask] = \
            sink_reg[:, mask]
        self.fields_reg_l[f, miny:maxy, minx:maxx][mask] = sink_l[mask]

        # update scale
        self.fields_width[f, miny:maxy, minx:maxx][mask] = np.log(width)
        self.fields_height[f, miny:maxy, minx:maxx][mask] = np.log(height)

    def fill_coordinate_kpsGradient(self, f, kps, width, height, scale):
        '''
        Use a wxh field but with weights obtained using a 2D Gaussian
        '''
        def gaussian2D(shape, sigma_x=1, sigma_y=1):
            w, h = shape
            if w == 1 and h == 1:
                return np.ones((1, 1))

            x = np.abs(np.linspace((w - 1.0) / 2.0, -(w - 1.0) / 2.0, num=w, dtype=np.float32))-0.5
            y = np.abs(np.linspace((h - 1.0) / 2.0, -(h - 1.0) / 2.0, num=h, dtype=np.float32))-0.5
            y = y.reshape(-1, 1)
            x = x.reshape(1,-1)

            re = np.exp(-(((x * x )/(2 * sigma_x * sigma_x)) + ((y * y)/(2 * sigma_y * sigma_y))))
            re[re < np.finfo(re.dtype).eps * re.max()] = 0
            return re

        xy_offset = [-0.5, -0.5]
        minx, miny = np.min(kps[kps[:,2]>0, 0]), np.min(kps[kps[:,2]>0, 1])
        maxx, maxy = np.max(kps[kps[:,2]>0, 0]), np.max(kps[kps[:,2]>0, 1])
        w = np.round(maxx - minx + 0.5).astype(np.int)
        w -= (w)%2
        h = np.round(maxy - miny + 0.5).astype(np.int)
        h -= (h)%2
        w = max(2,w)
        h = max(2, h)

        xyv = kps[-1]

        xy_offset = [(w - 1.0) / 2.0, (h - 1.0) / 2.0]

        ij = np.round(xyv[:2] - xy_offset).astype(np.int) + self.padding
        offset = xyv[:2] - (ij + xy_offset - self.padding)
        minx, miny = int(ij[0]), int(ij[1])

        if w<=0:
            raise Exception('w= ', w, ' is negative or zero')
        if h<=0:
            raise Exception('h= ', h, ' is negative or zero')

        maxx, maxy = minx + w, miny + h

        sink = create_sink_2d(w, h)
        gaussian = gaussian2D((w, h), sigma_x=w/4, sigma_y=(h)/4)
        assert sink.shape[1] == gaussian.shape[0]
        assert sink.shape[2] == gaussian.shape[1]
        if minx + w/2 < 0 or maxx - w/2 > self.intensities.shape[2] or \
           miny + h/2 < 0 or maxy - h/2 > self.intensities.shape[1]:
            return

        minx_n = max(0, minx)
        miny_n = max(0, miny)
        maxx_n = min(maxx, self.intensities.shape[2])
        maxy_n = min(maxy, self.intensities.shape[1])
        sink = sink[:, (miny_n-miny):(miny_n-miny) + (maxy_n-miny_n), (minx_n-minx):(minx_n-minx) + (maxx_n-minx_n)]
        gaussian = gaussian[(miny_n-miny):(miny_n-miny) + (maxy_n-miny_n), (minx_n-minx):(minx_n-minx) + (maxx_n-minx_n)]
        minx = minx_n
        maxx = maxx_n
        miny = miny_n
        maxy = maxy_n

        offset = offset.reshape(2, 1, 1)

        # update intensity
        masked_intensities = self.intensities[f, miny:maxy, minx:maxx]
        np.maximum(masked_intensities, gaussian, out=masked_intensities)
        # update regression
        sink_reg = sink + offset
        sink_l = np.linalg.norm(sink_reg, axis=0)
        mask = sink_l < self.fields_reg_l[f, miny:maxy, minx:maxx]
        self.fields_reg[f, :, miny:maxy, minx:maxx][:, mask] = \
            sink_reg[:, mask]
        self.fields_reg_l[f, miny:maxy, minx:maxx][mask] = sink_l[mask]

        # update scale
        self.fields_width[f, miny:maxy, minx:maxx][mask] = np.log(width)
        self.fields_height[f, miny:maxy, minx:maxx][mask] = np.log(height)

    def fill_coordinate_max4(self, f, kps, width, height, scale):
        '''
        Use a 4x4 field with ignore region surrounding it.
        '''
        xy_offset = [-0.5, -0.5]
        #xy_offset = [0, 0]
        minx, miny = np.min(kps[kps[:,2]>0, 0]), np.min(kps[kps[:,2]>0, 1])
        maxx, maxy = np.max(kps[kps[:,2]>0, 0]), np.max(kps[kps[:,2]>0, 1])
        #w = np.round(w - xy_offset[0]).astype(np.int)
        #h = np.round(h - xy_offset[1]).astype(np.int)
        w = np.round(maxx - minx + 0.5).astype(np.int)

        h = np.round(maxy- miny + 0.5).astype(np.int)

        xyv = kps[-1]

        xy_offset = [(4 - 1.0) / 2.0, (4 - 1.0) / 2.0]

        ij = np.round(xyv[:2] - xy_offset).astype(np.int) + self.padding
        offset = xyv[:2] - (ij + xy_offset - self.padding)
        minx, miny = int(ij[0]), int(ij[1])
        maxx, maxy = minx + 4, miny + 4

        if minx + 2 < 0 or maxx - 2 > self.intensities.shape[2] or \
           miny + 2 < 0 or maxy - 2 > self.intensities.shape[1]:
            return

        if w > 16 or h>16:
            sigma_ignore, sigma_eff = 0.5, 0.2
            w_ignore, h_ignore = np.round(sigma_ignore*w + 0.5).astype(np.int), np.round(sigma_ignore*h+ 0.5).astype(np.int)
            xy_offset = [(w_ignore - 1.0) / 2.0, (h_ignore - 1.0) / 2.0]
            ij = np.round(xyv[:2] - xy_offset).astype(np.int) + self.padding
            if w>16:
                minx_ignore = int(ij[0])
            else:
                minx_ignore = minx
                w_ignore = 4
            if h>16:
                miny_ignore = int(ij[1])
            else:
                miny_ignore = miny
                h_ignore = 4
            maxx_ignore, maxy_ignore = minx_ignore + w_ignore, miny_ignore + h_ignore
            self.intensities[f, miny_ignore:maxy_ignore, minx_ignore:maxx_ignore] = np.nan
        w = 4
        h = 4

        sink = create_sink_2d(w, h)

        minx_n = max(0, minx)
        miny_n = max(0, miny)
        maxx_n = min(maxx, self.intensities.shape[2])
        maxy_n = min(maxy, self.intensities.shape[1])
        sink = sink[:, (miny_n-miny):(miny_n-miny) + (maxy_n-miny_n), (minx_n-minx):(minx_n-minx) + (maxx_n-minx_n)]
        minx = minx_n
        maxx = maxx_n
        miny = miny_n
        maxy = maxy_n
        offset = offset.reshape(2, 1, 1)

        # update intensity
        self.intensities[f, miny:maxy, minx:maxx] = 1.0
        # update regression
        sink_reg = sink + offset
        sink_l = np.linalg.norm(sink_reg, axis=0)
        mask = sink_l < self.fields_reg_l[f, miny:maxy, minx:maxx]
        self.fields_reg[f, :, miny:maxy, minx:maxx][:, mask] = \
            sink_reg[:, mask]
        self.fields_reg_l[f, miny:maxy, minx:maxx][mask] = sink_l[mask]

        # update scale
        self.fields_width[f, miny:maxy, minx:maxx][mask] = np.log(width)
        self.fields_height[f, miny:maxy, minx:maxx][mask] = np.log(height)


    def fields(self, valid_area):
        intensities = self.intensities[:, self.padding:-self.padding, self.padding:-self.padding]
        fields_reg = self.fields_reg[:, :, self.padding:-self.padding, self.padding:-self.padding]
        fields_width = self.fields_width[:, self.padding:-self.padding, self.padding:-self.padding]
        fields_height = self.fields_height[:, self.padding:-self.padding, self.padding:-self.padding]
        #fields_scale = self.fields_scale[:, self.padding:-self.padding, self.padding:-self.padding]

        mask_valid_area(intensities, valid_area)

        return (
            torch.from_numpy(intensities),
            torch.from_numpy(fields_reg),
            torch.from_numpy(fields_width),
            torch.from_numpy(fields_height),
            #torch.from_numpy(fields_scale),
        )
