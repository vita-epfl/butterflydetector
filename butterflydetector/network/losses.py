"""Losses."""

import logging
import torch
import math
import numpy as np
from ..decoder.utils import index_field

LOG = logging.getLogger(__name__)



#@torch.jit.script
def ratio_iou_scripted(x1, y1, w1, h1, x2, y2, w2, h2):
    x1 = x1 - w1/2
    y1 = y1 - h1/2
    x2 = x2 - w2/2
    y2 = y2 - h2/2
    xi = torch.max(x1, x2)                                    # Intersection (yi similarly)
    yi = torch.max(y1, y2)                                    # Intersection (yi similarly)
    wi = torch.clamp(torch.min(x1+w1, x2+w2) - xi + 1, min=0, max=math.inf)
    hi = torch.clamp(torch.min(y1+h1, y2+h2) - yi + 1, min=0, max=math.inf)
    area_i = wi * hi                                      # Area Intersection
    area_u = (w1+1) * (h1+1) + (w2+1) * (h2+1) - wi * hi    # Area Union
    result = area_i / torch.clamp(area_u, min=1e-5, max=math.inf)
    try:
        assert((result>=0.).all() and (result<=1.0).all())
    except:
        import pdb; pdb.set_trace()
    return torch.clamp(result, min=1e-5, max=math.inf)

def ratio_siou_scripted(x1, y1, w1, h1, x2, y2, w2, h2):
    x1 = x1 - w1/2
    y1 = y1 - h1/2
    x2 = x2 - w2/2
    y2 = y2 - h2/2
    xi = torch.max(x1, x2)                                    # Intersection (yi similarly)
    yi = torch.max(y1, y2)
    wi = torch.min(x1+w1, x2+w2) - xi
    hi = torch.min(y1+h1, y2+h2) - yi
    mask = (wi>0) & (hi>0)
    area_i = - torch.abs(wi * hi)                                  # Area Intersection
    area_i[mask] = - area_i[mask]
    try:
        assert((w1>=0.).all() and (w2>=0).all() and (h1>=0).all() and (h2>=0).all())
    except:
        import pdb; pdb.set_trace()
    area_u = w1 * h1 + w2 * h2 - area_i    # Area Union
    result = area_i / area_u
    return result

def laplace_loss(x1, x2, logb, t1, t2, weight=None):
    """Loss based on Laplace Distribution.

    Loss for a single two-dimensional vector (x1, x2) with radial
    spread b and true (t1, t2) vector.
    """

    # left derivative of sqrt at zero is not defined, so prefer torch.norm():
    # https://github.com/pytorch/pytorch/issues/2421
    # norm = torch.sqrt((x1 - t1)**2 + (x2 - t2)**2)
    norm = (torch.stack((x1, x2)) - torch.stack((t1, t2))).norm(dim=0)

    losses = 0.694 + logb + norm * torch.exp(-logb)
    if weight is not None:
        losses = losses * weight
    return torch.sum(losses)


def l1_loss(x1, x2, _, t1, t2, weight=None):
    """L1 loss.

    Loss for a single two-dimensional vector (x1, x2)
    true (t1, t2) vector.
    """
    losses = torch.sqrt((x1 - t1)**2 + (x2 - t2)**2)
    if weight is not None:
        losses = losses * weight
    return torch.sum(losses)


def margin_loss(x1, x2, t1, t2, max_r1, max_r2, max_r3, max_r4):
    x = torch.stack((x1, x2))
    t = torch.stack((t1, t2))

    max_r = torch.min((torch.stack(max_r1, max_r2, max_r3, max_r4)), axis=0)
    m0 = torch.isfinite(max_r)
    x = x[:, m0]
    t = t[:, m0]
    max_r = max_r[m0]

    # m1 = (x - t).norm(p=1, dim=0) > max_r
    # x = x[:, m1]
    # t = t[:, m1]
    # max_r = max_r[m1]

    norm = (x - t).norm(dim=0)
    m2 = norm > max_r

    return torch.sum(norm[m2] - max_r[m2])


def quadrant(xys):
    q = torch.zeros((xys.shape[1],), dtype=torch.long)
    q[xys[0, :] < 0.0] += 1
    q[xys[1, :] < 0.0] += 2
    return q


def quadrant_margin_loss(x1, x2, t1, t2, max_r1, max_r2, max_r3, max_r4):
    x = torch.stack((x1, x2))
    t = torch.stack((t1, t2))

    diffs = x - t
    qs = quadrant(diffs)
    norms = diffs.norm(dim=0)

    m1 = norms[qs == 0] > max_r1[qs == 0]
    m2 = norms[qs == 1] > max_r2[qs == 1]
    m3 = norms[qs == 2] > max_r3[qs == 2]
    m4 = norms[qs == 3] > max_r4[qs == 3]

    return (
        torch.sum(norms[qs == 0][m1] - max_r1[qs == 0][m1]) +
        torch.sum(norms[qs == 1][m2] - max_r2[qs == 1][m2]) +
        torch.sum(norms[qs == 2][m3] - max_r3[qs == 2][m3]) +
        torch.sum(norms[qs == 3][m4] - max_r4[qs == 3][m4])
    )


class SmoothL1Loss(object):
    def __init__(self, r_smooth, scale_required=True):
        self.r_smooth = r_smooth
        self.scale = None
        self.scale_required = scale_required

    def __call__(self, x1, x2, _, t1, t2, weight=None):
        """L1 loss.

        Loss for a single two-dimensional vector (x1, x2)
        true (t1, t2) vector.
        """
        if self.scale_required and self.scale is None:
            raise Exception
        if self.scale is None:
            self.scale = 1.0

        r = self.r_smooth * self.scale
        d = torch.sqrt((x1 - t1)**2 + (x2 - t2)**2)
        smooth_regime = d < r

        smooth_loss = 0.5 / r[smooth_regime] * d[smooth_regime] ** 2
        linear_loss = d[smooth_regime == 0] - (0.5 * r[smooth_regime == 0])
        losses = torch.cat((smooth_loss, linear_loss))

        if weight is not None:
            losses = losses * weight

        self.scale = None
        return torch.sum(losses)

class MultiHeadLoss(torch.nn.Module):
    def __init__(self, losses, lambdas):
        super(MultiHeadLoss, self).__init__()

        self.losses = torch.nn.ModuleList(losses)
        self.lambdas = lambdas

        self.field_names = [n for l in self.losses for n in l.field_names]
        LOG.info('multihead loss: %s, %s', self.field_names, self.lambdas)

    def forward(self, head_fields, head_targets):  # pylint: disable=arguments-differ
        assert len(self.losses) == len(head_fields)
        assert len(self.losses) <= len(head_targets)
        flat_head_losses = [ll
                            for l, f, t in zip(self.losses, head_fields, head_targets)
                            for ll in l(f, t)]

        assert len(self.lambdas) == len(flat_head_losses)
        loss_values = [lam * l
                       for lam, l in zip(self.lambdas, flat_head_losses)
                       if l is not None]
        total_loss = sum(loss_values) if loss_values else None

        return total_loss, flat_head_losses

class MultiHeadLossAutoTune(torch.nn.Module):
    def __init__(self, losses, lambdas):
        """Auto-tuning multi-head less.

        Uses idea from "Multi-Task Learning Using Uncertainty to Weigh Losses
        for Scene Geometry and Semantics" by Kendall, Gal and Cipolla.

        In the common setting, use lambdas of zero and one to deactivate and
        activate the tasks you want to train. Less common, if you have
        secondary tasks, you can reduce their importance by choosing a
        lambda value between zero and one.
        """
        super().__init__()
        assert all(l >= 0.0 for l in lambdas)

        self.losses = torch.nn.ModuleList(losses)
        self.lambdas = lambdas
        self.log_sigmas = torch.nn.Parameter(
            torch.zeros((len(lambdas),), dtype=torch.float32),
            requires_grad=True,
        )

        self.field_names = [n for l in self.losses for n in l.field_names]
        LOG.info('multihead loss with autotune: %s', self.field_names)

    def batch_meta(self):
        return {'mtl_sigmas': [round(float(s), 3) for s in self.log_sigmas.exp()]}

    def forward(self, *args):
        head_fields, head_targets = args
        assert len(self.losses) == len(head_fields)
        assert len(self.losses) <= len(head_targets)
        flat_head_losses = [ll
                            for l, f, t in zip(self.losses, head_fields, head_targets)
                            for ll in l(f, t)]

        assert len(self.log_sigmas) == len(flat_head_losses)
        loss_values = np.array([lam * l / (2.0 * (log_sigma.exp() ** 2))
                       for lam, log_sigma, l in zip(self.lambdas, self.log_sigmas, flat_head_losses)
                       if l is not None])
        auto_reg = [lam * log_sigma
                    for lam, log_sigma, l in zip(self.lambdas, self.log_sigmas, flat_head_losses)
                    if l is not None]
        total_loss = sum(loss_values) + sum(auto_reg) if not(loss_values is None) else None

        return total_loss, flat_head_losses

class RepulsionLoss(torch.nn.Module):

    def __init__(self, use_gpu=True, sigma=0.):
        super(RepulsionLoss, self).__init__()

    def calc_iou_pairwise(self, a, b):
        area = b[:, 2] * b[:, 3]

        iw = torch.min(torch.unsqueeze((a[:, 0] + a[:, 2]), dim=1), (b[:, 0] + b[:, 2])) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
        ih = torch.min(torch.unsqueeze((a[:, 1] + a[:, 3]), dim=1), (b[:, 1] + b[:, 3])) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

        iw = torch.clamp(iw, min=0)
        ih = torch.clamp(ih, min=0)

        ua = torch.clamp(torch.unsqueeze(a[:, 2] * a[:, 3], dim=1) + area - iw * ih , min=1e-8)

        return (iw * ih) / ua

    def IoG(self, box_a, box_b):
        """Compute the IoG of two sets of boxes.
        E.g.:
            A ∩ B / A = A ∩ B / area(A)
        Args:
            box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
            box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_objects,4]
        Return:
            IoG: (tensor) Shape: [num_objects]
        """
        inter_xmin = torch.max(box_a[:, 0], box_b[:, 0])
        inter_ymin = torch.max(box_a[:, 1], box_b[:, 1])
        inter_xmax = torch.min((box_a[:, 0] + box_a[:, 2]), (box_b[:, 0] + box_b[:, 2]))
        inter_ymax = torch.min((box_a[:, 1] + box_a[:, 3]), (box_b[:, 1] + box_b[:, 3]))
        Iw = torch.clamp(inter_xmax - inter_xmin, min=0)
        Ih = torch.clamp(inter_ymax - inter_ymin, min=0)
        return (Iw * Ih) / (box_a[:, 2] * box_a[:, 3])

    # TODO
    def smooth_ln(self, x, smooth):
        return torch.where(
            torch.le(x, smooth),
            -torch.log(1 - x),
            ((x - smooth) / (1 - smooth)) - np.log(1 - smooth)
        )

    def forward(self, predict_boxes, ground_data, ids):

        RepGT_loss, RepBox_loss = torch.tensor(0).float().cuda(), torch.tensor(0).float().cuda()

        IoU = self.calc_iou_pairwise(ground_data, predict_boxes)
        IoU_argmax = torch.arange(len(ground_data)).cuda()
        positive_indices = torch.ge(IoU[IoU_argmax, IoU_argmax], 0.5)

        if positive_indices.sum() > 0:
            for index, elem in enumerate(ids):
                IoU[index, ids==elem] = -1
            _, IoU_argsec = torch.max(IoU, dim=1)
            IoG_to_minimize = self.IoG(ground_data[IoU_argsec, :], predict_boxes)
            RepGT_loss = self.smooth_ln(IoG_to_minimize, 0.5)
            RepGT_loss = RepGT_loss.mean()

            # add PepBox losses
            IoU_argmax_pos = IoU_argmax[positive_indices].float()
            IoU_argmax_pos = IoU_argmax_pos.unsqueeze(0).t()
            predict_boxes = torch.cat([predict_boxes, IoU_argmax_pos], dim=1)
            predict_boxes_np = predict_boxes.detach().cpu().numpy()
            num_gt = bbox_annotation.shape[0]
            predict_boxes_sampled = []
            for id in range(num_gt):
                index = np.where(predict_boxes_np[:, 4]==id)[0]
                if index.shape[0]:
                    idx = random.choice(range(index.shape[0]))
                    predict_boxes_sampled.append(predict_boxes[index[idx], :4])
            iou_repbox = self.calc_iou(predict_boxes_sampled, predict_boxes_sampled)
            iou_repbox = iou_repbox * mask
            RepBox_loss = self.smooth_ln(iou_repbox, 0.5)
            RepBox_loss = RepBox_loss.sum() / torch.clamp(torch.sum(torch.gt(iou_repbox, 0)).float(), min=1.0)
        return RepGT_loss, RepBox_loss

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.eps = eps
        self.gamma = gamma
        self.m = torch.nn.Sigmoid()
        self.previous=None

    def forward(self, preds, gt, bce_weight):
        pos_mask = gt.long().eq(1)
        neg_mask = gt.long().lt(1)
        res = self.m(preds)
        #res = torch.exp(res)
        #neg_weights = torch.pow(1 - gt[neg_mask], 4)
        pos_loss = self.alpha*torch.log(res[pos_mask]) * torch.pow(1 - res[pos_mask], self.gamma)
        neg_loss = (1-self.alpha)*torch.log(1 - res[neg_mask]) * torch.pow(res[neg_mask], self.gamma)#*neg_weights
        loss = torch.cat((pos_loss, pos_loss), 0)

        #self.previous = preds.clone()
        # if (gt==1).sum() != 0:
        #     return -self.alpha*loss/(gt==1).sum()
        # else:
        #     return -self.alpha*loss/1
        return -loss.mean()
        #import pdb; pdb.set_trace()
        #return torch.nn.functional.nll_loss(((1 - res) ** self.gamma) * log_res, gt.long(), bce_weight)

class CompositeLoss(torch.nn.Module):
    background_weight = 1.0
    multiplicity_correction = False
    independence_scale = 3.0

    def __init__(self, head_name, regression_loss, *,
                 n_vectors, n_scales, sigmas=None, margin=False, iou_loss=0, focal_loss=False):
        super(CompositeLoss, self).__init__()

        self.focal_loss = focal_loss
        if self.focal_loss:
            self.focal = FocalLoss(alpha=0.25, gamma=2, eps=1e-7)
        self.n_vectors = n_vectors
        self.n_scales = n_scales
        self.iou_loss = iou_loss
        if not ('butterfly' in head_name or 'repulse' in head_name):
            self.scales_butterfly = None
        if self.n_scales and not ('butterfly' in head_name or 'repulse' in head_name):
            assert len(sigmas) == n_scales
        elif self.n_scales:
            scales_butterfly = [[1.0] for _ in range(2)]
            scales_butterfly = torch.tensor(scales_butterfly)
            scales_butterfly = torch.unsqueeze(scales_butterfly, -1)
            scales_butterfly = torch.unsqueeze(scales_butterfly, -1)
            scales_butterfly = torch.unsqueeze(scales_butterfly, -1)
            self.register_buffer('scales_butterfly', scales_butterfly)

        if sigmas is None:
            sigmas = [[1.0] for _ in range(n_vectors)]
        if sigmas is not None and 'butterfly' in head_name:
            assert len(sigmas) == n_vectors
            scales_to_kp = torch.tensor(sigmas)
            scales_to_kp = torch.unsqueeze(scales_to_kp, 0)
            scales_to_kp = torch.unsqueeze(scales_to_kp, -1)
            scales_to_kp = torch.unsqueeze(scales_to_kp, -1)
            self.register_buffer('scales_to_kp', scales_to_kp)
        else:
            self.scales_to_kp = None

        self.regression_loss = regression_loss or laplace_loss

        if self.iou_loss>0:
            self.field_names = (
                ['{}.c'.format(head_name)] +
                ['{}.vec{}'.format(head_name, i + 1) for i in range(self.n_vectors)] +
                #['{}.scales{}'.format(head_name, i + 1) for i in range(self.n_scales)] +
                ['{}.iou{}'.format(head_name, i + 1) for i in range(self.n_vectors)]
            )
        else:
            self.field_names = (
                ['{}.c'.format(head_name)] +
                ['{}.vec{}'.format(head_name, i + 1) for i in range(self.n_vectors)] +
                ['{}.scales{}'.format(head_name, i + 1) for i in range(self.n_scales)]
            )
        self.margin = margin
        if self.margin:
            self.field_names += ['{}.margin{}'.format(head_name, i + 1)
                                 for i in range(self.n_vectors)]

        self.bce_blackout = None

        self.repulsion_loss = None
        if 'repulse' in head_name:
            self.repulsion_loss = RepulsionLoss()
        LOG.debug('%s: n_vectors = %d, n_scales = %d, len(sigmas) = %d, margin = %s',
                  head_name, n_vectors, n_scales, len(sigmas), margin)
    def forward(self, *args):  # pylint: disable=too-many-statements
        LOG.debug('loss for %s', self.field_names)

        x, t = args

        assert len(x) == 1 + 2 * self.n_vectors + self.n_scales
        x_intensity = x[0]
        x_regs = x[1:1 + self.n_vectors]
        x_spreads = x[1 + self.n_vectors:1 + 2 * self.n_vectors]
        x_scales = []
        if self.n_scales:
            x_scales = x[1 + 2 * self.n_vectors:1 + 2 * self.n_vectors + self.n_scales]

        #assert len(t) == 1 + self.n_vectors + self.n_scales
        target_intensity = t[0]
        target_regs = t[1:1 + self.n_vectors]
        target_scales = t[1 + self.n_vectors:1 + self.n_vectors + self.n_scales]
        bce_masks = (target_intensity[:, :-1] + target_intensity[:, -1:]) > 0.5
        if not torch.any(bce_masks):
            return None, None, None

        batch_size = x_intensity.shape[0]
        LOG.debug('batch size = %d', batch_size)

        bce_x_intensity = x_intensity
        bce_target_intensity = target_intensity[:, :-1]
        if self.bce_blackout:
            bce_x_intensity = bce_x_intensity[:, self.bce_blackout]
            bce_masks = bce_masks[:, self.bce_blackout]
            bce_target_intensity = bce_target_intensity[:, self.bce_blackout]

        LOG.debug('BCE: x = %s, target = %s, mask = %s',
                  x_intensity.shape, bce_target_intensity.shape, bce_masks.shape)
        # bce_masks = (
        #     bce_masks
        #     & ((x_intensity > -4.0) | ((x_intensity < -4.0) & (target_intensity[:, :-1] == 1)))
        #     & ((x_intensity < 4.0) | ((x_intensity > 4.0) & (target_intensity[:, :-1] == 0)))
        # )
        bce_target = torch.masked_select(bce_target_intensity, bce_masks)
        bce_weight = None
        if self.background_weight != 1.0:
            bce_weight = torch.ones_like(bce_target)
            bce_weight[bce_target == 0] = self.background_weight

        if self.focal_loss:
            ce_loss = self.focal(torch.masked_select(bce_x_intensity, bce_masks), bce_target, bce_weight)#/batch_size
        else:
            ce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                torch.masked_select(bce_x_intensity, bce_masks),
                bce_target,
                reduction='sum',
            ) / 100.0 / batch_size
        reg_losses = [None for _ in target_regs]
        reg_masks = target_intensity[:, :-1] > 0.5
        if torch.any(reg_masks):
            weight = None
            if self.multiplicity_correction:
                assert len(target_regs) == 2
                lengths = torch.norm(target_regs[0] - target_regs[1], dim=2)
                multiplicity = (lengths - 3.0) / self.independence_scale
                multiplicity = torch.clamp(multiplicity, min=1.0)
                multiplicity = torch.masked_select(multiplicity, reg_masks)
                weight = 1.0 / multiplicity

            reg_losses = []
            if self.iou_loss<3:
                for i, (x_reg, x_spread, target_reg) in enumerate(zip(x_regs, x_spreads, target_regs)):
                    if hasattr(self.regression_loss, 'scale'):
                        assert self.scales_to_kp is not None
                        self.regression_loss.scale = torch.masked_select(
                            torch.clamp(target_scales[i], 0.1, 1000.0),  # pylint: disable=unsubscriptable-object
                            reg_masks,
                        )

                    reg_losses.append(self.regression_loss(
                        torch.masked_select(x_reg[:, :, 0], reg_masks),
                        torch.masked_select(x_reg[:, :, 1], reg_masks),
                        torch.masked_select(x_spread, reg_masks),
                        torch.masked_select(target_reg[:, :, 0], reg_masks),
                        torch.masked_select(target_reg[:, :, 1], reg_masks),
                        weight=(weight if weight is not None else 1.0) * 0.1,
                    ) / 100.0 / batch_size)

        scale_losses = []
        wh_losses = []
        if self.iou_loss==0:
            if not self.scales_butterfly is None:
                if x_scales:
                    wh_losses = [
                        torch.nn.functional.l1_loss(
                            torch.masked_select(x_scale, torch.isnan(target_wh) == 0),
                            torch.masked_select(target_wh, torch.isnan(target_wh) == 0),
                            reduction='sum',
                        ) / 1000.0 / batch_size
                        for x_scale, target_wh, scale_to_kp in zip(x_scales, target_scales, self.scales_butterfly)
                    ]
            else:
                if x_scales:
                    scale_losses = [
                        torch.nn.functional.l1_loss(
                            torch.masked_select(x_scale, torch.isnan(target_scale) == 0),
                            torch.masked_select(target_scale, torch.isnan(target_scale) == 0),
                            reduction='sum',
                        ) / 1000.0 / batch_size
                        for x_scale, target_scale, scale_to_kp in zip(x_scales, target_scales, self.scales_to_kp)
                    ]

        iou_losses = []
        repgt_losses = []
        repbbox_losses = []
        if self.iou_loss>0 or self.repulsion_loss:
            import pdb; pdb.set_trace()
            if self.repulsion_loss:
                fields_ids = t[-1]
            for i, (x_reg, target_reg) in enumerate(zip(x_regs, target_regs)):
                index_fields = index_field(x_reg[:, :, 0:2].shape[-2:])
                index_fields = np.expand_dims(index_fields, 0)
                index_fields = np.expand_dims(index_fields, 0)
                joint_fields_pred = torch.from_numpy(index_fields.copy()).cuda() + x_reg[:, :, 0:2]

                joint_fields_gt = torch.from_numpy(index_fields.copy()).cuda() + target_reg[:, :, 0:2]


                w_pred = torch.exp(torch.masked_select(x_scales[0], reg_masks))
                h_pred = torch.exp(torch.masked_select(x_scales[1], reg_masks))
                x_pred = torch.masked_select(joint_fields_pred[:, :, 0], reg_masks)
                y_pred = torch.masked_select(joint_fields_pred[:, :, 1], reg_masks)


                w_gt = torch.exp(torch.masked_select(target_scales[0], reg_masks))
                h_gt = torch.exp(torch.masked_select(target_scales[1], reg_masks))
                x_gt = torch.masked_select(joint_fields_gt[:,:,0], reg_masks)
                y_gt = torch.masked_select(joint_fields_gt[:,:,1], reg_masks)
                if self.repulsion_loss:
                    x_gt = x_gt - w_gt/2
                    y_gt = y_gt - h_gt/2
                    x_pred = x_pred - w_pred/2
                    y_pred = y_pred - h_pred/2
                    bbox_pred = torch.cat((torch.unsqueeze(x_pred,1), torch.unsqueeze(y_pred,1), torch.unsqueeze(w_pred,1), torch.unsqueeze(h_pred,1)), 1)
                    bbox_gt = torch.cat((torch.unsqueeze(x_gt,1), torch.unsqueeze(y_gt,1), torch.unsqueeze(w_gt,1), torch.unsqueeze(h_gt,1)), 1)
                    repgt_loss, repbbox_loss = self.repulsion_loss(bbox_pred, bbox_gt, ids = torch.masked_select(fields_ids, reg_masks))
                    repgt_losses.append(repgt_loss)
                    repbbox_losses.append(repbbox_loss)

                else:
                    if self.iou_loss%2 != 0:
                        iou_pred = ratio_iou_scripted(x_pred, y_pred, \
                                                      w_pred, h_pred, \
                                                      x_gt, y_gt, \
                                                      w_gt, h_gt)
                        iou_loss = torch.nn.functional.binary_cross_entropy(
                            iou_pred,
                            torch.ones_like(iou_pred),
                            reduction='sum',
                        )/ 1000.0/ batch_size
                    elif self.iou_loss != 0:
                        iou_pred = ratio_siou_scripted(x_pred, y_pred, \
                                                      w_pred, h_pred, \
                                                      x_gt, y_gt, \
                                                      w_gt, h_gt)

                        iou_loss = (1 - iou_pred).sum()/1000.0/ batch_size
                    iou_losses.append(iou_loss)


        margin_losses = [None for _ in target_regs] if self.margin else []
        if self.margin and torch.any(reg_masks):
            margin_losses = []
            for i, (x_reg, target_reg) in enumerate(zip(x_regs, target_regs)):
                margin_losses.append(quadrant_margin_loss(
                    torch.masked_select(x_reg[:, :, 0], reg_masks),
                    torch.masked_select(x_reg[:, :, 1], reg_masks),
                    torch.masked_select(target_reg[:, :, 0], reg_masks),
                    torch.masked_select(target_reg[:, :, 1], reg_masks),
                    torch.masked_select(target_reg[:, :, 2], reg_masks),
                    torch.masked_select(target_reg[:, :, 3], reg_masks),
                    torch.masked_select(target_reg[:, :, 4], reg_masks),
                    torch.masked_select(target_reg[:, :, 5], reg_masks),
                ) / 100.0 / batch_size)
        return [ce_loss] + reg_losses + scale_losses + margin_losses + wh_losses + iou_losses + repgt_losses + repbbox_losses

def cli(parser):
    group = parser.add_argument_group('losses')
    group.add_argument('--lambdas', default=[30.0, 2.0, 2.0, 50.0, 3.0, 3.0],
                       type=float, nargs='+',
                       help='prefactor for head losses')
    group.add_argument('--r-smooth', type=float, default=0.0,
                       help='r_{smooth} for SmoothL1 regressions')
    group.add_argument('--regression-loss', default='laplace',
                       choices=['smoothl1', 'smootherl1', 'l1', 'laplace', 'laplace_iou', 'laplace_siou', 'iou_only', 'siou_only', 'laplace_focal'],
                       help='type of regression loss')
    group.add_argument('--background-weight', default=1.0, type=float,
                       help='[experimental] BCE weight of background')
    group.add_argument('--margin-loss', default=False, action='store_true',
                       help='[experimental]')
    group.add_argument('--auto-tune-mtl', default=False, action='store_true',
                       help='[experimental]')


def factory_from_args(args):
    # apply for CompositeLoss
    CompositeLoss.background_weight = args.background_weight

    return factory(
        args.headnets,
        args.lambdas,
        reg_loss_name=args.regression_loss,
        r_smooth=args.r_smooth,
        device=args.device,
        margin=args.margin_loss,
        auto_tune_mtl=args.auto_tune_mtl,
    )


def loss_parameters(head_name):
    n_vectors = 1

    n_scales = 2
    return {
        'n_vectors': n_vectors,
        'n_scales': n_scales,
    }


def factory(head_names, lambdas, *,
            reg_loss_name=None, r_smooth=None, device=None, margin=False,
            auto_tune_mtl=False):
    if isinstance(head_names[0], (list, tuple)):
        return [factory(hn, lam,
                        reg_loss_name=reg_loss_name,
                        r_smooth=r_smooth,
                        device=device,
                        margin=margin)
                for hn, lam in zip(head_names, lambdas)]

    head_names = [h for h in head_names if h not in ('skeleton', 'tskeleton')]
    iou_loss = 0
    if reg_loss_name == 'smoothl1':
        reg_loss = SmoothL1Loss(r_smooth)
    elif reg_loss_name == 'l1':
        reg_loss = l1_loss
    elif reg_loss_name == 'laplace' or reg_loss_name == 'laplace_focal':
        reg_loss = laplace_loss
    elif reg_loss_name == 'laplace_iou':
        reg_loss = laplace_loss
        iou_loss = 1
    elif reg_loss_name == 'laplace_siou':
        reg_loss = laplace_loss
        iou_loss = 2
    elif reg_loss_name == 'iou_only':
        reg_loss = laplace_loss
        iou_loss = 3
    elif reg_loss_name == 'siou_only':
        reg_loss = laplace_loss
        iou_loss = 4
    elif reg_loss_name is None:
        reg_loss = laplace_loss
    else:
        raise Exception('unknown regression loss type {}'.format(reg_loss_name))

    losses = [CompositeLoss(head_name, reg_loss,
                            margin=margin, iou_loss=iou_loss, focal_loss='focal' in reg_loss_name, **loss_parameters(head_name))
              for head_name in head_names]
    if auto_tune_mtl:
        loss = MultiHeadLossAutoTune(losses, lambdas)
    else:
        loss = MultiHeadLoss(losses, lambdas)

    if device is not None:
        loss = loss.to(device)

    return loss
