from contextlib import contextmanager

import numpy as np
from PIL import Image

try:
    import matplotlib
    import matplotlib.collections
    import matplotlib.pyplot as plt
    import matplotlib.patches
except ImportError:
    matplotlib = None
    plt = None


@contextmanager
def canvas(fig_file=None, show=True, dpi=200, **kwargs):
    if 'figsize' not in kwargs:
        # kwargs['figsize'] = (15, 8)
        kwargs['figsize'] = (10, 6)
    fig, ax = plt.subplots(**kwargs)

    yield ax

    fig.set_tight_layout(True)
    if fig_file:
        fig.savefig(fig_file, dpi=dpi)  # , bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


@contextmanager
def image_canvas(image, fig_file=None, show=True, dpi_factor=1.0, fig_width=10.0, **kwargs):
    image = np.asarray(image)
    if 'figsize' not in kwargs:
        kwargs['figsize'] = (fig_width, fig_width * image.shape[0] / image.shape[1])

    fig = plt.figure(**kwargs)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    fig.add_axes(ax)
    ax.imshow(image)

    yield ax

    if fig_file:
        fig.savefig(fig_file, dpi=image.shape[1] / kwargs['figsize'][0] * dpi_factor)
    if show:
        plt.show()
    plt.close(fig)


def load_image(path, scale=1.0):
    with open(path, 'rb') as f:
        image = Image.open(f).convert('RGB')
        image = np.asarray(image) * scale / 255.0
        return image


class CrowdPainter(object):
    def __init__(self, *, alpha=0.5, color='orange'):
        self.alpha = alpha
        self.color = color

    def draw(self, ax, outlines):
        for outline in outlines:
            assert outline.shape[1] == 2

        patches = []
        for outline in outlines:
            polygon = matplotlib.patches.Polygon(
                outline[:, :2], color=self.color, facecolor=self.color, alpha=self.alpha)
            patches.append(polygon)
        ax.add_collection(matplotlib.collections.PatchCollection(patches, match_original=True))


class InstancePainter(object):
    def __init__(self, *,
                 xy_scale=1.0, highlight=None, highlight_invisible=False,
                 show_box=False,
                 show_joint_scale=False,
                 show_decoding_order=False,
                 linewidth=2, markersize=3,
                 color_connections=False,
                 solid_threshold=0.5):
        self.xy_scale = xy_scale
        self.highlight = highlight
        self.highlight_invisible = highlight_invisible
        self.show_box = show_box
        self.show_joint_scale = show_joint_scale
        self.show_decoding_order = show_decoding_order
        self.linewidth = linewidth
        self.markersize = markersize
        self.color_connections = color_connections
        self.solid_threshold = solid_threshold

    @staticmethod
    def _draw_box(ax, x1, y1, x2, y2, v, color, score=None):
        if not np.any(v > 0):
            return

        # keypoint bounding box
        #x1, x2 = np.min(x[v > 0]), np.max(x[v > 0])
        #y1, y2 = np.min(y[v > 0]), np.max(y[v > 0])
        if x2 - x1 < 5.0:
            x1 -= 2.0
            x2 += 2.0
        if y2 - y1 < 5.0:
            y1 -= 2.0
            y2 += 2.0
        ax.add_patch(
            matplotlib.patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, fill=False, color=color))

        if score:
            ax.text(x1, y1, '{:.4f}'.format(score), fontsize=8, color=color)

    @staticmethod
    def _draw_text(ax, x, y, v, text, color):

        ax.text(x, y, text, fontsize=8,
                color='white', bbox={'facecolor': color, 'alpha': 0.5, 'linewidth': 0})

    @staticmethod
    def _draw_scales(ax, xs, ys, vs, color, scales):
        for x, y, v, scale in zip(xs, ys, vs, scales):
            if v == 0.0:
                continue
            ax.add_patch(
                matplotlib.patches.Rectangle(
                    (x - scale, y - scale), 2 * scale, 2 * scale, fill=False, color=color))

    def output_butterfly(self, ax, keypoint_sets, bboxes, *, scores=None, color=None, colors=None, texts=None):

        if keypoint_sets is None:
            return

        if color is None and self.color_connections:
            color = 'white'
        if color is None and colors is None:
            colors = range(len(keypoint_sets))
        for i, (kps, bbox) in enumerate(zip(np.asarray(keypoint_sets),np.asarray(bboxes))):
            assert kps.shape[1] == 3
            v = kps[:, 2]
            if not np.any(v > 0):
                continue
            mask = v > 0
            v = kps[mask, 2]
            x = kps[mask, 0] * self.xy_scale
            y = kps[mask, 1] * self.xy_scale
            x1 = bbox[mask, 0] * self.xy_scale
            y1 = bbox[mask, 1] * self.xy_scale
            w = bbox[mask, 2] * self.xy_scale
            h = bbox[mask, 3] * self.xy_scale
            category = np.argmax(keypoint_sets[i,:,2], axis=0)

            if colors is not None:
                color = colors[category]

            if isinstance(color, (int, np.integer)):
                color = matplotlib.cm.get_cmap('tab20')((color % 20 + 0.05) / 20)

            if w < 5.0:
                x1 -= 2.0
                w += 2.0
            if h < 5.0:
                y1 -= 2.0
                h += 2.0
            ax.add_patch(
                matplotlib.patches.Rectangle(
                    (x1, y1), w, h, fill=False, color=color))
            circle = matplotlib.patches.Circle(
                (x1+w/2, (y1+h/2)), max(w,h)/ 100.0, zorder=10, linewidth=1)
            ax.add_patch(circle)
            score = scores[i] if scores is not None else None
            ax.text(x1, y1, '{:.4f}'.format(score), fontsize=8, color=color)
            if self.show_box:
                score = scores[i] if scores is not None else None
                self._draw_box(ax, x, y, v, color, score)

            if texts is not None:
                self._draw_text(ax, x, y, v, texts[i], color)

    def annotations(self, ax, annotations, *,
                    color=None, colors=None, scores=None, texts=None):
        if annotations is None:
            return

        if color is None and self.color_connections:
            color = 'white'
        if color is None and colors is None:
            colors = range(len(annotations))

        for i, ann in enumerate(annotations):
            if colors is not None:
                color = colors[i]
            if scores is not None:
                score = scores[i]

            text = texts[i] if texts is not None else None
            self.annotation(ax, ann, color=color, score=score, text=text)

    def annotation(self, ax, ann, *, color, score=None, text=None):
        if isinstance(color, (int, np.integer)):
            color = matplotlib.cm.get_cmap('tab20')((color % 20 + 0.05) / 20)

        kps = ann.data[:, :3]
        v = kps[:, 2]
        if not np.any(v > 0):
            return
        mask = v > 0
        bbox = np.array([ann.data[:, 0]-ann.data[:, 3]/2, ann.data[:, 1]-ann.data[:, 4]/2, ann.data[:, 3], ann.data[:, 4]]).T
        assert kps.shape[1] == 3
        x = kps[mask, 0] * self.xy_scale
        y = kps[mask, 1] * self.xy_scale
        x1 = bbox[mask, 0] * self.xy_scale
        y1 = bbox[mask, 1] * self.xy_scale
        w = bbox[mask, 2] * self.xy_scale
        h = bbox[mask, 3] * self.xy_scale
        v = kps[mask, 2]
        category = np.argmax(ann.data[:,2], axis=0)
        if w < 5.0:
            x1 -= 2.0
            w += 2.0
        if h < 5.0:
            y1 -= 2.0
            h += 2.0

        self._draw_box(ax, x1, y1, x1+w, y1+h, v, color, ann.score())

        if text is not None:
            self._draw_text(ax, x, y, v, text, color)

    @staticmethod
    def _draw_decoding_order(ax, decoding_order):
        for step_i, (jsi, jti, jsxyv, jtxyv) in enumerate(decoding_order):
            ax.plot([jsxyv[0], jtxyv[0]], [jsxyv[1], jtxyv[1]], '--', color='black')
            ax.text(0.5 * (jsxyv[0] + jtxyv[0]), 0.5 * (jsxyv[1] +jtxyv[1]),
                    '{}: {} -> {}'.format(step_i, jsi, jti), fontsize=8,
                    color='white', bbox={'facecolor': 'black', 'alpha': 0.5, 'linewidth': 0})


def quiver(ax, vector_field, intensity_field=None, step=1, threshold=0.5,
           xy_scale=1.0, uv_is_offset=False,
           reg_uncertainty=None, **kwargs):
    x, y, u, v, c, r = [], [], [], [], [], []
    for j in range(0, vector_field.shape[1], step):
        for i in range(0, vector_field.shape[2], step):
            if intensity_field is not None and intensity_field[j, i] < threshold:
                continue
            x.append(i * xy_scale)
            y.append(j * xy_scale)
            u.append(vector_field[0, j, i] * xy_scale)
            v.append(vector_field[1, j, i] * xy_scale)
            c.append(intensity_field[j, i] if intensity_field is not None else 1.0)
            r.append(reg_uncertainty[j, i] * xy_scale if reg_uncertainty is not None else None)
    x = np.array(x)
    y = np.array(y)
    u = np.array(u)
    v = np.array(v)
    c = np.array(c)
    r = np.array(r)
    s = np.argsort(c)
    if uv_is_offset:
        u -= x
        v -= y

    for xx, yy, uu, vv, _, rr in zip(x, y, u, v, c, r):
        if not rr:
            continue
        circle = matplotlib.patches.Circle(
            (xx + uu, yy + vv), rr / 2.0, zorder=11, linewidth=1, alpha=1.0,
            fill=False, color='orange')
        ax.add_artist(circle)

    return ax.quiver(x[s], y[s], u[s], v[s], c[s],
                     angles='xy', scale_units='xy', scale=1, zOrder=10, **kwargs)


def margins(ax, vector_field, intensity_field=None, step=1, threshold=0.5,
            xy_scale=1.0, uv_is_offset=False, **kwargs):
    x, y, u, v, r = [], [], [], [], []
    for j in range(0, vector_field.shape[1], step):
        for i in range(0, vector_field.shape[2], step):
            if intensity_field is not None and intensity_field[j, i] < threshold:
                continue
            x.append(i * xy_scale)
            y.append(j * xy_scale)
            u.append(vector_field[0, j, i] * xy_scale)
            v.append(vector_field[1, j, i] * xy_scale)
            r.append(vector_field[2:6, j, i] * xy_scale)
    x = np.array(x)
    y = np.array(y)
    u = np.array(u)
    v = np.array(v)
    r = np.array(r)
    if uv_is_offset:
        u -= x
        v -= y

    wedge_angles = [
        (0.0, 90.0),
        (90.0, 180.0),
        (270.0, 360.0),
        (180.0, 270.0),
    ]

    for xx, yy, uu, vv, rr in zip(x, y, u, v, r):
        for q_rr, (theta1, theta2) in zip(rr, wedge_angles):
            if not np.isfinite(q_rr):
                continue
            wedge = matplotlib.patches.Wedge(
                (xx + uu, yy + vv), q_rr, theta1, theta2,
                zorder=9, linewidth=1, alpha=0.5 / 16.0,
                fill=True, color='orange', **kwargs)
            ax.add_artist(wedge)


def arrows(ax, fourd, xy_scale=1.0, threshold=0.0, **kwargs):
    mask = np.min(fourd[:, 2], axis=0) >= threshold
    fourd = fourd[:, :, mask]
    (x1, y1), (x2, y2) = fourd[:, :2, :] * xy_scale
    c = np.min(fourd[:, 2], axis=0)
    s = np.argsort(c)
    return ax.quiver(x1[s], y1[s], (x2 - x1)[s], (y2 - y1)[s], c[s],
                     angles='xy', scale_units='xy', scale=1, zOrder=10, **kwargs)


def boxes(ax, scalar_field, intensity_field=None, xy_scale=1.0, step=1, threshold=0.5,
          cmap='viridis_r', clim=(0.5, 1.0), **kwargs):
    x, y, s, c = [], [], [], []
    for j in range(0, scalar_field.shape[0], step):
        for i in range(0, scalar_field.shape[1], step):
            if intensity_field is not None and intensity_field[j, i] < threshold:
                continue
            x.append(i * xy_scale)
            y.append(j * xy_scale)
            s.append(scalar_field[j, i] * xy_scale)
            c.append(intensity_field[j, i] if intensity_field is not None else 1.0)

    cmap = matplotlib.cm.get_cmap(cmap)
    cnorm = matplotlib.colors.Normalize(vmin=clim[0], vmax=clim[1])
    for xx, yy, ss, cc in zip(x, y, s, c):
        color = cmap(cnorm(cc))
        rectangle = matplotlib.patches.Rectangle(
            (xx - ss, yy - ss), ss * 2.0, ss * 2.0,
            color=color, zorder=10, linewidth=1, **kwargs)
        ax.add_artist(rectangle)

def boxes_butterfly(ax, width_field, height_field, reg_fields=None, intensity_field=None, xy_scale=1.0, step=1, threshold=0.1,
          cmap='viridis_r', clim=(0.5, 1.0), **kwargs):
    x, y, w, h, o_x, o_y, c = [], [], [], [], [], [], []
    for j in range(0, width_field.shape[0], step):
        for i in range(0, width_field.shape[1], step):
            if intensity_field is not None and intensity_field[j, i] < threshold:
                continue
            x.append(i * xy_scale)
            y.append(j * xy_scale)
            if not reg_fields is None:
                o_x.append(reg_fields[0][j, i]* xy_scale)
                o_y.append(reg_fields[1][j, i]* xy_scale)
                w.append(np.exp(width_field[j, i]) * xy_scale)
                h.append(np.exp(height_field[j, i]) * xy_scale)
            else:
                o_x.append(0)
                o_y.append(0)
                w.append(width_field[j, i] * xy_scale)
                h.append(height_field[j, i] * xy_scale)
            c.append(intensity_field[j, i] if intensity_field is not None else 1.0)

    cmap = matplotlib.cm.get_cmap(cmap)
    cnorm = matplotlib.colors.Normalize(vmin=clim[0], vmax=clim[1])
    for xx, yy, ww, hh, oo_x, oo_y, cc in zip(x, y, w, h, o_x, o_y, c):
        color = cmap(cnorm(cc))
        rectangle = matplotlib.patches.Rectangle(
            (xx+oo_x - ww/2, yy+oo_y - hh/2), ww, hh,
            color=color, zorder=10, linewidth=1, **kwargs)
        ax.add_artist(rectangle)


def circles(ax, scalar_field, intensity_field=None, xy_scale=1.0, step=1, threshold=0.5,
            cmap='viridis_r', clim=(0.5, 1.0), **kwargs):
    x, y, s, c = [], [], [], []
    for j in range(0, scalar_field.shape[0], step):
        for i in range(0, scalar_field.shape[1], step):
            if intensity_field is not None and intensity_field[j, i] < threshold:
                continue
            x.append(i * xy_scale)
            y.append(j * xy_scale)
            s.append(scalar_field[j, i] * xy_scale)
            c.append(intensity_field[j, i] if intensity_field is not None else 1.0)

    cmap = matplotlib.cm.get_cmap(cmap)
    cnorm = matplotlib.colors.Normalize(vmin=clim[0], vmax=clim[1])
    for xx, yy, ss, cc in zip(x, y, s, c):
        color = cmap(cnorm(cc))
        circle = matplotlib.patches.Circle(
            (xx, yy), ss,
            color=color, zorder=10, linewidth=1, **kwargs)
        ax.add_artist(circle)


def white_screen(ax, alpha=0.9):
    ax.add_patch(
        plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, alpha=alpha,
                      facecolor='white')
    )
