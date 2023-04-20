import logging
import numpy as np
import re

from ..data import BBOX_POINTS, BBOX_SKELETON_COMPLETE
from .. import show

LOG = logging.getLogger(__name__)


class Visualizer(object):
    def __init__(self, head_names, strides, *,
                 file_prefix=None, fields_indices=None,
                 show_margin=False, show_plots=False):
        self.head_names = head_names
        self.strides = strides
        self.fields_indices = fields_indices or []
        self.show_margin = show_margin
        self.file_prefix = file_prefix
        self.show = show_plots
        self.keypoint_painter = show.InstancePainter(show_box=True)

    def single(self, image, targets, meta):
        keypoint_sets = None
        if 'skeleton' in self.head_names:
            i = self.head_names.index('skeleton')
            keypoint_sets = targets[i][0]

        print('Image file name', meta['file_name'])
        with show.canvas() as ax:
            ax.imshow(image)

        for target, headname, stride in zip(targets, self.head_names, self.strides):
            LOG.debug('%s with %d components', headname, len(target))
            if 'butterfly' in headname:
                scale_wh = stride
                if 'nsbutterfly' in headname:
                    scale_wh = 1
                self.butterfly(image, target, meta, stride, scale_wh, keypoint_sets, keypoints=BBOX_POINTS)
            elif 'repulse' in headname:
                scale_wh = stride
                if 'nsrepulse' in headname:
                    scale_wh = 1
                self.butterfly_repulse(image, target, stride, scale_wh, keypoint_sets, keypoints=BBOX_POINTS)
            else:
                LOG.warning('unknown head: %s', headname)

    # def tile(self,a, dim, n_tile):
    #     init_dim = a.shape(dim)
    #     repeat_idx = [1] * a.dim()
    #     repeat_idx[dim] = n_tile
    #     a = a.repeat(*(repeat_idx))
    #     order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    #     return torch.index_select(a, dim, order_index)

    def butterfly(self, image, target, meta, stride, scale_wh, keypoint_sets, *, keypoints):
        import os
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            from mpl_toolkits.axes_grid1 import make_axes_locatable
        except ImportError:
            matplotlib = None
        resized_image = image[::stride, ::stride]
        bce_targets = target[0]
        bce_masks = (bce_targets[:-1] + bce_targets[-1:]) > 0.5
        if -1 in self.fields_indices:
            fields_indices = np.arange(bce_targets.shape[0]-1)[np.nanmax(bce_targets[:-1], axis=(1,2))==1]
        else:
            fields_indices = self.fields_indices
        filename = os.path.splitext(os.path.basename(meta['file_name']))[0]
        fig_file = os.path.join(self.file_prefix, filename + '.butterfly.bkgrnd.png') if self.file_prefix else None
        with show.canvas(fig_file, show=self.show) as ax:
            ax.imshow(image)
            ax.imshow(np.repeat(np.repeat(bce_targets[-1],stride, axis=1), stride, axis=0)[:image.shape[1], :image.shape[0]], alpha=0.9, vmin=0.0, vmax=1.0)

        for f in fields_indices:
            print('intensity field', keypoints[-1])

            bbox_masks = bce_targets[f]==1
            fig_file = os.path.join(self.file_prefix, filename + '.butterfly{}.c.png'.format(f)) if self.file_prefix else None
            with show.canvas(fig_file, show=self.show) as ax:
                #ax.imshow(resized_image)
                #ax.imshow(target[0][f] + 0.5 * bce_masks[f], alpha=0.2, vmin=0.0, vmax=1.0)
                ax.imshow(image)
                im = ax.imshow(np.repeat(np.repeat(target[0][f] + 0.5 * bce_masks[f],stride, axis=1), stride, axis=0)[:image.shape[1], :image.shape[0]], alpha=0.9,
                               vmin=0.0, vmax=1.0, cmap='YlOrRd')

                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='3%', pad=0.05)
                plt.colorbar(im, cax=cax)

            fig_file = os.path.join(self.file_prefix, filename + '.butterfly{}.wh.png'.format(f)) if self.file_prefix else None
            with show.canvas(fig_file, show=self.show) as ax:
                ax.imshow(image)
                for iy,ix in np.ndindex(target[0][f].shape):
                    if not bbox_masks[iy,ix]:
                        continue
                    w = np.exp(target[2][f][iy,ix]) * scale_wh
                    h = np.exp(target[3][f][iy,ix]) * scale_wh
                    cx = (ix + target[1][f][0][iy,ix]) * stride
                    cy = (iy + target[1][f][1][iy,ix]) * stride
                    x1 = cx - w/2
                    y1 = cy - h/2
                    ax.add_patch(
                        matplotlib.patches.Rectangle(
                            (x1, y1), w, h, fill=False, color=None))
                    circle = matplotlib.patches.Circle(
                        (cx, cy), max(w,h)/ 10.0, zorder=10, linewidth=1)
                    ax.add_patch(circle)

            fig_file = os.path.join(self.file_prefix, filename + '.butterfly{}.v.png'.format(f)) if self.file_prefix else None
            with show.canvas(fig_file, show=self.show) as ax:
                ax.imshow(image)
                show.white_screen(ax, alpha=0.5)
                # self.keypoint_painter.output_butterfly(ax, keypoint_sets[:, (f*5):f*5+5])
                show.quiver(ax, target[1][f], xy_scale=stride)

    def butterfly_repulse(self, image, target, stride, scale_wh, keypoint_sets, *, keypoints):
        try:
            import matplotlib
        except ImportError:
            matplotlib = None
        resized_image = image[::stride, ::stride]
        bce_targets = target[0]
        bce_masks = (bce_targets[:-1] + bce_targets[-1:]) > 0.5

        for f in self.fields_indices:
            print('intensity field', keypoints[-1])

            bbox_masks = bce_targets[f]==1
            with show.canvas() as ax:
                ax.imshow(resized_image)
                ax.imshow(target[0][f] + 0.5 * bce_masks[f], alpha=0.9, vmin=0.0, vmax=1.0)

            with show.canvas() as ax:
                ax.imshow(image)
                for iy,ix in np.ndindex(target[0][f].shape):
                    if not bbox_masks[iy,ix]:
                        continue
                    w = np.exp(target[2][f][iy,ix]) * scale_wh
                    h = np.exp(target[3][f][iy,ix]) * scale_wh
                    cx = (ix + target[1][f][0][iy,ix]) * stride
                    cy = (iy + target[1][f][1][iy,ix]) * stride
                    x1 = cx - w/2
                    y1 = cy - h/2
                    ax.add_patch(
                        matplotlib.patches.Rectangle(
                            (x1, y1), w, h, fill=False, color=None))
                    circle = matplotlib.patches.Circle(
                        (cx, cy), max(w,h)/ 10.0, zorder=10, linewidth=1)
                    ax.add_patch(circle)

            with show.canvas() as ax:
                ax.imshow(resized_image)
                ax.imshow(target[4][f] + bce_masks[f]*1, alpha=0.9, vmin=0.0, cmap='tab20')

            with show.canvas() as ax:
                ax.imshow(image)
                show.white_screen(ax, alpha=0.5)
                self.keypoint_painter.output_butterfly(ax, keypoint_sets[:, (f*5):f*5+5])
                show.quiver(ax, target[1][f], xy_scale=stride)

    def __call__(self, images, targets, meta):
        n_heads = len(targets)
        n_batch = len(images)
        targets = [[t.numpy() for t in heads] for heads in targets]
        targets = [
            [[target_field[batch_i] for target_field in targets[head_i]]
             for head_i in range(n_heads)]
            for batch_i in range(n_batch)
        ]

        images = np.moveaxis(np.asarray(images), 1, -1)
        images = np.clip((images + 2.0) / 4.0, 0.0, 1.0)

        for i, t, m in zip(images, targets, meta):
            self.single(i, t, m)
