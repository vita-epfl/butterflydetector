import numpy as np

# pylint: disable=import-error
from ..functional import scalar_value_clipped

class Annotation(object):
    def __init__(self, j, xyv, n_joints, dim_per_kps=3):
        self.data = np.zeros((n_joints, dim_per_kps))
        self.joint_scales_w = None
        self.joint_scales_h = None
        self.data[j] = xyv

    def fill_joint_scales(self, scales_w, scales_h, hr_scale=1.0):
        self.joint_scales_w = np.zeros((self.data.shape[0],))
        self.joint_scales_h = np.zeros((self.data.shape[0],))
        for xyv_i, xyv in enumerate(self.data):
            if xyv[2] == 0.0:
                continue
            scale_field_w = scales_w[xyv_i]
            scale_field_h = scales_h[xyv_i]
            #i = max(0, min(scale_field_w.shape[1] - 1, int(round(xyv[0] * hr_scale))))
            #j = max(0, min(scale_field_w.shape[0] - 1, int(round(xyv[1] * hr_scale))))
            scale_w = scalar_value_clipped(scale_field_w, xyv[0] * hr_scale, xyv[1] * hr_scale)
            scale_h = scalar_value_clipped(scale_field_h, xyv[0] * hr_scale, xyv[1] * hr_scale)
            self.joint_scales_w[xyv_i] = scale_w / hr_scale
            self.joint_scales_h[xyv_i] = scale_h / hr_scale

    def fill_joint_scales_nothr(self, scales_w, scales_h, index, hr_scale=1.0):
        self.joint_scales_w = np.zeros((self.data.shape[0],))
        self.joint_scales_h = np.zeros((self.data.shape[0],))
        for xyv_i, xyv in enumerate(self.data):
            if xyv[2] == 0.0:
                continue
            self.joint_scales_w[xyv_i] = scales_w[xyv_i][index] / hr_scale
            self.joint_scales_h[xyv_i] = scales_h[xyv_i][index] / hr_scale

    def score(self):
        v = self.data[:, 2]
        return np.max(v)
        # return np.mean(np.square(v))

    def scale(self):
        m = self.data[:, 2] > 0.5
        if not np.any(m):
            return 0.0
        return max(
            np.max(self.data[m, 0]) - np.min(self.data[m, 0]),
            np.max(self.data[m, 1]) - np.min(self.data[m, 1]),
        )
